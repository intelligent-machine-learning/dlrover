# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Dict

from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    JobExitReason,
    NodeType,
    OptimizeMode,
    PlatformType,
    RendezvousName,
    ReporterType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.diagnosis.diagnosis import DiagnosisManager
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
    RendezvousManager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.master import JobMaster
from dlrover.python.master.monitor.error_monitor import ErrorMonitor
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.node.event_callback import (
    AllReduceNodeHandlingCallback,
    TaskRescheduleCallback,
    TFPSNodeHandlingCallback,
)
from dlrover.python.master.servicer import create_master_service
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.scheduler.job import JobArgs


def _create_elastic_ps_service_if_needed(params: JobArgs):
    if params.distribution_strategy == DistributionStrategy.PS:
        return ElasticPsService()
    return None


def _create_master_service_on_k8s(namespace, job_name, job_uuid, target_port):
    from dlrover.python.scheduler.kubernetes import (
        NODE_SERVICE_PORTS,
        k8sClient,
        k8sServiceFactory,
    )

    owner_ref = k8sClient.create_owner_reference(
        api_version="elastic.iml.github.io/v1alpha1",
        kind="ElasticJob",
        name=job_name,
        uid=job_uuid,
    )

    svc_factory = k8sServiceFactory(namespace, job_name)
    svc_name = f"elasticjob-{job_name}-dlrover-master"
    port = NODE_SERVICE_PORTS[NodeType.DLROVER_MASTER]
    selector = {
        ElasticJobLabel.JOB_KEY: job_name,
        ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.DLROVER_MASTER,
    }
    succeed = svc_factory.create_service(
        name=svc_name,
        port=port,
        target_port=target_port,
        selector=selector,
        owner_ref=owner_ref,
    )
    return succeed


class DistributedJobMaster(JobMaster):
    """The master of a distrbiuted job which has multiple nodes. The master
    - launches nodes (e.g. the Pod on kubernetes).
    - builds the rendezvous of training ndoes.
    - monitors the node status and launch a new node to recover a failed node.
    - collects the training metrics including throughput and the workload
        of each node.
    - auto-scales nodes of a job to speed up the training and improve the
        resource utilization.

    The master mainly contains the following components:
    JobManager: manages the nodes of a job. the job manager can launch nodes,
        monitor nodes and scale up/down nodes.
    RendezvousManager: build the rendezvous of training nodes.
    TaskManager: assignes the data shard tasks to workers and recover the data
        shard task of a failed worker.
    MetricCollector: collects the training metrics of a training job.
    ElasticPSService: manages the hosts of alive PS nodes in a PS training job.
    """

    def __init__(
        self, port, args: JobArgs, error_monitor: ErrorMonitor = None
    ):
        if args.platform in [
            PlatformType.KUBERNETES,
            PlatformType.PY_KUBERNETES,
        ]:
            succeed = _create_master_service_on_k8s(
                args.namespace, args.job_name, args.job_uuid, port
            )
            if not succeed:
                logger.warning(
                    "Fail to create the master service. "
                    "The master cannot recover from the failure."
                )

        self.speed_monitor = SpeedMonitor()
        self.job_manager = (
            create_job_manager(args, self.speed_monitor)
            if args.enable_elastic_scheduling
            else None
        )
        self.task_manager = (
            TaskManager(
                args.node_args[NodeType.WORKER].restart_timeout,
                self.speed_monitor,
            )
            if args.enable_dynamic_sharding
            else None
        )
        elastic_training = RendezvousName.ELASTIC_TRAINING
        self.rdzv_managers: Dict[str, RendezvousManager] = {
            elastic_training: ElasticTrainingRendezvousManager(error_monitor),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(
                error_monitor
            ),
        }
        self.diagnosis_manager = DiagnosisManager()
        self.job_metric_collector = self._create_metric_collector_if_needed(
            args
        )
        self.elastic_ps_service = _create_elastic_ps_service_if_needed(args)
        self.sync_service = SyncService(self.job_manager)
        self._master_server = self._create_master_grpc_service(port, args)
        self._job_args = args
        self._stop_requested = False
        self._exit_code = 0
        self._exit_reason = None

    def _create_master_grpc_service(self, port, params: JobArgs):
        return create_master_service(
            port,
            self.task_manager,
            self.job_manager,
            self.speed_monitor,
            self.rdzv_managers,
            self.diagnosis_manager,
            self.job_metric_collector,
            self.elastic_ps_service,
            self.sync_service,
        )

    def _create_metric_collector_if_needed(self, params: JobArgs):
        if not params.enable_dynamic_sharding:
            return None
        job_uuid = params.job_uuid
        reporter = ReporterType.LOCAL
        if params.optimize_mode == OptimizeMode.CLUSTER:
            reporter = ReporterType.DLROVER_BRAIN
        collector = JobMetricCollector(
            job_uuid, params.namespace, params.cluster, params.user, reporter
        )
        collector.collect_job_type(params.distribution_strategy)
        return collector

    def prepare(self):
        # Start the master GRPC server
        logger.info("Starting master RPC server")
        self._master_server.start()
        logger.info("Master RPC server started")

        # Composite the components
        if self.task_manager and self.job_manager:
            self.task_manager.set_task_timeout_callback(
                self.job_manager.remove_worker
            )
        if self.job_manager:
            self._add_node_event_callback()

        # Start the components one by one
        if self.task_manager:
            self.task_manager.start()
        if self.job_manager:
            self.job_manager.start()

    def pre_check(self):
        logger.info("Pre-check before running.")
        self.diagnosis_manager.pre_check()
        # TODO

    def _add_node_event_callback(self):
        """Add NodeEventCallbacks for the listeners of Pod events."""
        if self.task_manager:
            self.job_manager.add_node_event_callback(
                TaskRescheduleCallback(self.task_manager)
            )
        strategy = self._job_args.distribution_strategy
        if strategy == DistributionStrategy.PS:
            self.job_manager.add_node_event_callback(
                TFPSNodeHandlingCallback(self)
            )
        elif strategy == DistributionStrategy.ALLREDUCE:
            self.job_manager.add_node_event_callback(
                AllReduceNodeHandlingCallback(self)
            )

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """

        # start training runtime diagnosis
        try:
            self.diagnosis_manager.start_observing()
        except Exception as e:
            logger.warning(
                "Failed to start training " f"runtime diagnosis: {str(e)}"
            )

        # into running loop
        try:
            while True:
                if self._stop_requested:
                    break
                should_stop, reason, msg = self.job_manager.should_early_stop()
                if should_stop:
                    self.request_stop(False, reason, msg)
                    continue
                self.job_manager.clear_exited_nodes()
                if self.job_manager and self.job_manager.all_workers_exited():
                    if self.job_manager.pend_without_workers():
                        time.sleep(30)
                        continue
                    if self.job_manager.all_workers_failed():
                        logger.error("All workers failed")
                        self._exit_code = 1
                        self._exit_reason = JobExitReason.UNKNOWN_ERROR
                    elif (
                        self.task_manager and not self.task_manager.finished()
                    ):
                        logger.warning(
                            "All workers exited but there also are "
                            "unfinished tasks",
                        )
                    break

                if (
                    self.job_manager.all_running_node_hanged()
                    and self.task_manager.task_hanged()
                ):
                    logger.error("All nodes hangeds")
                    self._exit_code = 1
                    self._exit_reason = JobExitReason.HANG_ERROR

                if (
                    self.task_manager
                    and self.task_manager.finished()
                    and (
                        not self.job_manager
                        or self.job_manager.all_critical_node_completed()
                    )
                ):
                    logger.info("All task completed")
                    break

                time.sleep(30)
        except KeyboardInterrupt:
            logger.warning("Server stopping")
        finally:
            if self.job_manager:
                self.job_manager.stop()
            if self.diagnosis_manager:
                self.diagnosis_manager.stop_observing()
            self.stop()

        return self._exit_code

    def _remove_not_participated_workers(self):
        """Remove workers who do not participate training."""
        for manager in self.rdzv_managers.values():
            ranks = manager.not_joined_rdzv_nodes()
            if ranks:
                self.job_manager.remove_not_joined_rdzv_workers(ranks)

    def stop(self):
        """
        Stop all the components.
        Make sure that the created services and components are shut down.
        """
        if self._exit_code == 0 and not self._exit_reason:
            self._exit_reason = JobExitReason.SUCCEEDED
        logger.info("Job exit with the reason {}".format(self._exit_reason))
        if self.job_metric_collector:
            self.job_metric_collector.collect_job_exit_reason(
                self._exit_reason
            )
        logger.info("Stopping master")
        logger.info("Stopping RPC server")
        self._master_server.stop(grace=None)
        logger.info("RPC server stopped")
        logger.info("Master stopped")

    def request_stop(self, success, reason, msg=""):
        self._stop_requested = True
        self._exit_reason = reason
        if success:
            self._exit_code = 0
            logger.info(
                f"Request to stop. Success: {success}, reason: {reason}, "
                f"msg: {msg}."
            )
        else:
            self._exit_code = 1
            logger.error(
                f"Request to stop. Success: {success}, reason: {reason}, "
                f"msg: {msg}."
            )
