# Copyright 2025 The DLRover Authors. All rights reserved.
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
from typing import Dict, List

import ray

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeType,
    PlatformType,
    PreCheckStatus,
    RendezvousName,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
    RendezvousManager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.servicer import RayMasterServicer
from dlrover.python.scheduler.job import JobArgs, NodeArgs
from dlrover.python.unified.common.enums import InternalRoleType
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.elastic.failover import (
    ElasticFailoverCoordinator,
)
from dlrover.python.unified.master.elastic.job_manager import ElasticJobManager
from dlrover.python.unified.master.master import BaseMaster


@ray.remote
class ElasticMaster(BaseMaster):
    """
    Elastic master is a control implementation designed for training scenarios
    involving single compute roles: using torch elastic training.
    """

    def __init__(self, job_config_serialized, dl_context_serialized):
        super().__init__(job_config_serialized, dl_context_serialized)

        # core components for torch elastic training management
        self._rdzv_managers = None
        self._perf_monitor = None
        self._job_manager = None
        self._sync_service = None
        self._diagnosis_manager = None
        self._master_service_handler = None
        self._failover_coordinator = None

        self.init()
        logger.info(f"Elastic master initialized: {self.job_name}.")

    def init(self):
        job_args = self._get_job_args_from_unified_context()

        # setup context
        # TODO: replaced by pre-check procedure(skip for now)
        get_job_context().set_pre_check_status(PreCheckStatus.DISABLED)

        # init core component
        self._rdzv_managers: Dict[str, RendezvousManager] = {
            RendezvousName.TRAINING: ElasticTrainingRendezvousManager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        self._perf_monitor = PerfMonitor()
        self._job_manager = ElasticJobManager()
        self._sync_service = SyncService(self._job_manager)
        self._diagnosis_manager = DiagnosisMaster(job_args)
        self._master_service_handler = RayMasterServicer(
            task_manager=None,  # no need
            job_manager=self._job_manager,
            perf_monitor=self._perf_monitor,
            rdzv_managers=self._rdzv_managers,
            diagnosis_manager=self._diagnosis_manager,
            job_metric_collector=None,  # no need
            elastic_ps_service=None,  # no need
            sync_service=self._sync_service,
        )
        self._failover_coordinator = ElasticFailoverCoordinator(
            self._job_manager, self._save_context_to_checkpoint, self.exit_job
        )

    def _get_job_args_from_unified_context(self):
        job_args = JobArgs(
            PlatformType.RAY, "default", self.context.job_config.job_name
        )
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        job_args.job_uuid = job_args.job_name

        node_args: Dict[str, NodeArgs] = {
            NodeType.WORKER: NodeArgs(
                None,  # no need
                auto_scale=False,
                restart_count=self.context.job_config.get_workload_max_restart(
                    InternalRoleType.ELASTIC.name
                ),
                critical_nodes="",
            )
        }
        job_args.node_args = node_args
        return job_args

    def _handle_failures(self, failures: List[FailureDesc]):
        self._failover_coordinator.handle_failures(failures)

    def _get_master_wait_interval(self):
        return 5

    """Remote call functions start"""

    def agent_report(self, request):
        logger.debug(f"Got agent report call: {request}")
        response = self._master_service_handler.agent_report(request)
        logger.debug(f"Response agent report call: {response}")
        return response

    def agent_get(self, request):
        logger.debug(f"Got agent get call: {request}")
        response = self._master_service_handler.agent_get(request)
        logger.debug(f"Response agent get call: {response}")
        return response

    """Remote call functions end"""
