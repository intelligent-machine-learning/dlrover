# Copyright 2026 The DLRover Authors. All rights reserved.
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
import threading
import time
from typing import Dict

from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    EventReportConstants,
    JobExitReason,
    NodeType,
    OptimizeMode,
    PlatformType,
    PreCheckStatus,
    RendezvousName,
    ReporterType,
)
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    JobAbortionAction,
    JobRestartAction,
)
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.elastic_training.rdzv_manager import (
    NetworkCheckRendezvousManager,
    RendezvousManager,
    create_training_rdzv_manager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.master import JobMaster, get_service_type
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.node.event_callback import (
    AllReduceNodeHandlingCallback,
    TaskRescheduleCallback,
    TFPSNodeHandlingCallback,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.servicer import create_master_service
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.watcher.factory import new_elasticjob_watcher
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.training_event import DLRoverMasterEvent
from dlrover.python.util.function_util import TimeoutException

_master_evt = DLRoverMasterEvent().singleton_instance()


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
    """The master of a distributed job which has multiple nodes. The master
    - launches nodes (e.g. the Pod on kubernetes).
    - builds the rendezvous of training nodes.
    - monitors the node status and launch a new node to recover a failed node.
    - collects the training metrics including throughput and the workload
        of each node.
    - auto-scales nodes of a job to speed up the training and improve the
        resource utilization.

    The master mainly contains the following components:
    JobManager: manages the nodes of a job. the job manager can launch nodes,
        monitor nodes and scale up/down nodes.
    RendezvousManager: build the rendezvous of training nodes.
    TaskManager: assigns the data shard tasks to workers and recover the data
        shard task of a failed worker.
    MetricCollector: collects the training metrics of a training job.
    ElasticPSService: manages the hosts of alive PS nodes in a PS training job.
    """

    def __init__(self, port, args: JobArgs):
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

        self._job_ctx = get_job_context()
        self.perf_monitor = PerfMonitor()
        self.job_manager = (
            create_job_manager(args, self.perf_monitor)
            if args.enable_elastic_scheduling
            else None
        )
        self.task_manager = (
            TaskManager(
                args.node_args[NodeType.WORKER].process_timeout,
                self.perf_monitor,
            )
            if args.enable_dynamic_sharding
            else None
        )
        self.rdzv_managers: Dict[str, RendezvousManager] = {
            RendezvousName.TRAINING: create_training_rdzv_manager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        self.diagnosis_manager = DiagnosisMaster(args)
        self._event_reporter = get_event_reporter()
        self.job_metric_collector = self._create_metric_collector_if_needed(
            args
        )
        self.elastic_ps_service = _create_elastic_ps_service_if_needed(args)
        self.sync_service = SyncService(self.job_manager)
        self._master_server = self._create_master_service(port, args)
        self._job_args = args
        self._job_evt = _master_evt.train_job(
            job_name=args.job_name, args=vars(args)
        )
        self._elasticjob_watcher = new_elasticjob_watcher(args)

    @property
    def exit_code(self):
        return self._job_ctx.get_exit_code()

    @property
    def exit_reason(self):
        return self._job_ctx.get_exit_reason()

    def set_exit(self, exit_code, exit_reason=""):
        self._job_ctx.set_exit_code(exit_code)
        self._job_ctx.set_exit_reason(exit_reason)

    def _create_master_service(self, port, params: JobArgs):
        return create_master_service(
            port,
            self.task_manager,
            self.job_manager,
            self.perf_monitor,
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
        # start the master server
        logger.info(f"Starting master {get_service_type()} server")
        self._master_server.start()
        logger.info(f"Master {get_service_type()} server started")

        if self._elasticjob_watcher:
            self._elasticjob_watcher.start()

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

        threading.Thread(
            target=self._diagnose_job,
            name="job_diagnosing",
            daemon=True,
        ).start()

    def _diagnose_job(self):
        logger.info("Start diagnosing the job.")
        while True:
            if self._job_ctx.is_stopped():
                logger.info("Stop diagnosing job.")
                break

            # deal with diagnosis action
            action = self._job_ctx.next_action(
                instance=DiagnosisConstant.MASTER_INSTANCE
            )

            if isinstance(action, JobAbortionAction):
                logger.info(f"Got job abortion action: {action}")
                self.request_stop(
                    success=False,
                    reason=action.reason,
                    msg=action.msg,
                )
            elif isinstance(action, JobRestartAction):
                if not self._job_ctx.is_restarting():
                    logger.warning(f"Got job restart action: {action}")
                    self.request_restart(action.reason, action.msg)
            else:
                self.job_manager.process_diagnosis_action(action)

            # 10 actions per second
            time.sleep(0.1)

    def pre_check(self):
        logger.info("Pre-check before running.")
        start = time.time()
        try:
            self.diagnosis_manager.pre_check()
            logger.info(
                f"Pre-check finished, cost: {time.time() - start:.2f}s."
            )
        except TimeoutException:
            logger.warning("Pre-check timeout, set pass as result for safety.")
            self._job_ctx.set_pre_check_status(PreCheckStatus.PASS)

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
            self.diagnosis_manager.start_metric_collect()
        except Exception as e:
            logger.warning(f"Failed to start metric collecting: {str(e)}")

        try:
            self.diagnosis_manager.start_observing()
        except Exception as e:
            logger.warning(
                f"Failed to start training runtime diagnosis: {str(e)}"
            )

        self._event_reporter.report_job_start(self._job_evt, self._job_args)

        # into running loop
        try:
            while True:
                if self._job_ctx.is_stopped():
                    logger.info(
                        f"Job is stopped: {self._job_ctx.get_job_stage()}"
                    )
                    break
                elif self._job_ctx.is_restarting():
                    logger.info("Trigger job restarting.")

                    # sync implement to restart job
                    self.job_manager.restart()

                should_stop, reason, msg = self.job_manager.should_early_stop()
                if should_stop:
                    self._job_evt.fail(
                        error=f"{reason}",
                        msg=msg,
                    )
                    self.request_stop(False, reason, msg)
                    break
                self.job_manager.clear_exited_nodes()
                if self.job_manager and self.job_manager.all_workers_exited():
                    if self.job_manager.pend_without_workers():
                        time.sleep(30)
                        continue
                    if self.job_manager.all_workers_failed():
                        logger.error("All workers failed")
                        self.set_exit(1, JobExitReason.UNKNOWN_ERROR)
                    elif (
                        self.task_manager
                        and not self.task_manager.finished()
                        and self.task_manager.is_dataset_initialized()
                    ):
                        logger.warning(
                            "All workers exited but there also are unfinished tasks",
                        )
                    break

                if (
                    self.job_manager.all_running_node_hanged()
                    and self.task_manager.task_hanged()
                ):
                    logger.error("All nodes hanged")
                    self.set_exit(1, JobExitReason.HANG_ERROR)

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
                self.diagnosis_manager.stop_metric_collect()
                self.diagnosis_manager.stop_observing()
            self.stop()

        if self.exit_code == 0:
            self._event_reporter.report_job_success(
                self._job_evt, self._job_args
            )
        else:
            self._event_reporter.report_job_fail(
                self._job_evt, self._job_args, self.exit_reason
            )

        return self.exit_code

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
        if self.exit_code == 0 and not self.exit_reason:
            self.set_exit(0, JobExitReason.SUCCEEDED)
        logger.info("Job exit with the reason {}".format(self.exit_reason))
        if self.job_metric_collector:
            self.job_metric_collector.collect_job_exit_reason(self.exit_reason)
        logger.info("Stopping master")
        logger.info("Stopping RPC server")
        self._master_server.stop(grace=None)
        logger.info("RPC server stopped")
        logger.info("Master stopped")

    def request_stop(self, success, reason, msg=""):
        if success:
            logger.info(
                f"Request to stop. Success: {success}, reason: {reason}, msg: {msg}."
            )
        else:
            logger.error(
                f"Request to stop. Success: {success}, reason: {reason}, msg: {msg}."
            )

        action = EventReportConstants.ACTION_STOP
        if not success:
            action = EventReportConstants.ACTION_EARLY_STOP
        if self._event_reporter:
            self._event_reporter.report(
                event_type=EventReportConstants.TYPE_ERROR,
                instance="job",
                action=action,
                msg=msg,
                labels={
                    "reason": reason,
                    "success": f"{success}",
                },
            )
        exit_code = 0 if success else 1
        self._job_ctx.request_stop(exit_code, reason)

    def request_restart(self, reason, msg=""):
        logger.info(f"Request to restart. Reason: {reason}, msg: {msg}.")

        if self._event_reporter:
            self._event_reporter.report(
                event_type=EventReportConstants.TYPE_WARN,
                instance="job",
                action=EventReportConstants.ACTION_RESTART,
                msg=msg,
                labels={
                    "reason": reason,
                },
            )
        self._job_ctx.request_restart()
