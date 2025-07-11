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
    NodeType,
    OptimizeMode,
    PreCheckStatus,
    RendezvousName,
    ReporterType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
    RendezvousManager,
)
from dlrover.python.master.master import JobMaster, get_service_type
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.local_job_manager import create_job_manager
from dlrover.python.master.servicer import create_master_service
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.scheduler.job import JobArgs


class LocalJobMaster(JobMaster):
    def __init__(self, port, args: JobArgs):
        self.perf_monitor = PerfMonitor()
        self.task_manager = TaskManager(0, self.perf_monitor)
        self.job_manager = create_job_manager(args, self.perf_monitor)
        self.diagnosis_manager = DiagnosisMaster(args)
        elastic_training = RendezvousName.TRAINING
        self.rdzv_managers: Dict[str, RendezvousManager] = {
            elastic_training: ElasticTrainingRendezvousManager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        self.job_metric_collector = self._create_metric_collector_if_needed(
            args
        )
        self._master_server = self._create_master_service(port, args)
        self._job_args = args
        for i in range(args.node_args[NodeType.WORKER].group_resource.count):
            self.perf_monitor.add_running_worker(NodeType.WORKER, i)
        self.perf_monitor.set_target_worker_num(1)

    def _create_master_service(self, port, params: JobArgs):
        return create_master_service(
            port,
            self.task_manager,
            self.job_manager,
            self.perf_monitor,
            self.rdzv_managers,
            self.diagnosis_manager,
            self.job_metric_collector,
            None,
            None,
        )

    def _create_metric_collector_if_needed(self, params: JobArgs):
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
        self.task_manager.start()
        self.job_manager.start()

    def pre_check(self):
        get_job_context().set_pre_check_status(PreCheckStatus.DISABLED)

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """
        try:
            while True:
                if self.task_manager and self.task_manager.finished():
                    logger.info("All task completed!")
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            logger.warning("Server stopping!")
        finally:
            self.stop()
        return 0

    def stop(self):
        """
        Stop all the components.
        Make sure that the created services and components are shut down.
        """
        logger.info("Stopping master!")
        logger.info("Stopping RPC server!")
        self._master_server.stop(grace=None)
        logger.info("RPC server stopped!")
        logger.info("Master stopped!")

    def request_stop(self, success, reason, msg=""):
        pass
