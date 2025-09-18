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


from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import ActorBase
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.decorators import log_execution

from .manager import ElasticManager


class ElasticMaster(ActorBase):
    def setup(self):
        workers = PrimeMasterApi.get_workers_by_role(self.actor_info.role)
        self.manager = ElasticManager(workers)

    # region Lifecycle Hooks

    async def check_workers(self):
        await self.manager.check_workers()

    async def start(self):
        with log_execution("ElasticMaster start"):
            await self.manager.start()

    def recover_running(self):
        # manager has nothing to recover
        pass

    def restart_role_level(self):
        self.manager.request_restart()

    def handle_worker_failover(self, worker_name: str) -> bool:
        logger.info(f"Worker {worker_name} needs failover, request restart.")
        # Elastic training currently only supports role-level restart.
        self.manager.request_restart()
        return True

    # region RPC methods for Workers
    # Currently, ElasticMaster as controller, no additional rpc methods are needed.

    # TODO metric rpc: AtorchEvent, Event
    # TODO diagnosis rpc: NodeFailure, ResourceStats, DiagnosisReportData(XPUTimer)
