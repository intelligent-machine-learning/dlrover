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

from dlrover.python.common.constants import (
    DistributionStrategy,
    JobExitReason,
    NodeType,
    OptimizeMode,
    RendezvousName,
    ReporterType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.event_callback import (
    AllReduceNodeHandlingCallback,
    TaskRescheduleCallback,
    TFPSNodeHandlingCallback,
)
from dlrover.python.master.node.job_manager import create_job_manager
from dlrover.python.master.servicer import create_master_service
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.scheduler.job import JobArgs


def _create_elastic_ps_service_if_needed(params: JobArgs):
    if params.distribution_strategy == DistributionStrategy.PS:
        return ElasticPsService()
    return None


class Master(object):
    def __init__(self, port, args: JobArgs):
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
        self.rdzv_managers = {
            elastic_training: ElasticTrainingRendezvousManager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
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
        try:
            while True:
                if self._stop_requested:
                    break
                self._remove_not_participated_workers()
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
                    self._exit_reason = JobExitReason.UNKNOWN_ERROR

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
            self.stop()

        return self._exit_code

    def _remove_not_participated_workers(self):
        """Remove workers who do not participate training."""
        et_manager = self.rdzv_managers[RendezvousName.ELASTIC_TRAINING]
        workers = et_manager.get_released_workers()
        if workers:
            self.job_manager.remove_not_participated_workers(workers)

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
            logger.info(msg)
        else:
            self._exit_code = 1
            logger.error(msg)
