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
from typing import Dict

import ray

from dlrover.python.common.constants import RendezvousName, PlatformType, \
    DistributionStrategy, NodeType
from dlrover.python.master.elastic_training.rdzv_manager import \
    RendezvousManager, ElasticTrainingRendezvousManager, \
    NetworkCheckRendezvousManager
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.servicer import create_master_service
from dlrover.python.scheduler.job import JobArgs, NodeArgs
from dlrover.python.unified.common.enums import InternalRoleType
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.master import BaseMaster
from dlrover.python.unified.master.spmd.job_manager import ElasticJobManager


@ray.remote
class SPMDMaster(BaseMaster):
    """
    SPMD master is a control implementation designed for training scenarios
    involving single compute roles: using torch elastic.
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

        self.init()

    def init(self):
        job_args = self._get_job_args_from_unified_context()

        self._rdzv_managers: Dict[str, RendezvousManager] = {
            RendezvousName.ELASTIC_TRAINING: ElasticTrainingRendezvousManager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        self._perf_monitor = PerfMonitor()
        self._job_manager = ElasticJobManager(job_args, self._perf_monitor, None)
        self._sync_service = SyncService(self._job_manager)
        self._master_service_handler = create_master_service(
            0,  # no need
            None, # no need
            self._job_manager,
            self._perf_monitor,
            self._rdzv_managers,
            self._diagnosis_manager,
            None, # no need
            None, # no need
            self._sync_service,
        )

    def _get_job_args_from_unified_context(self):
        job_args = JobArgs(PlatformType.RAY, "default", self.context.job_config.job_name)
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        job_args.job_uuid = job_args.job_name

        node_args: Dict[str, NodeArgs] = {
            NodeType.WORKER: NodeArgs(None,  # no need
                                      auto_scale=False,
                                      restart_count=self.context.job_config.get_workload_max_restart(InternalRoleType.ELASTIC.name),
                                      critical_nodes="")
        }
        job_args.node_args = node_args
        return job_args

    def _handle_failure(self, failure: FailureDesc):
        pass

    """Remote call functions start"""

    def agent_report(self, request):
        return self._master_service_handler.agent_report(request)

    def agent_get(self, request):
        return self._master_service_handler.agent_get(request)

    """Remote call functions end"""
