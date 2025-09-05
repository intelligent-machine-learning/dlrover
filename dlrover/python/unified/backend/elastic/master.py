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


import ray

from dlrover.python.unified.common.actor_base import ActorBase
from dlrover.python.unified.controller.api import PrimeMasterApi

from .manager import ElasticManager


class ElasticMaster(ActorBase):
    def _setup(self):
        workers = PrimeMasterApi.get_workers_by_role(self.actor_info.role)
        self.manager = ElasticManager(workers)
        if ray.get_runtime_context().was_current_actor_reconstructed():
            self.manager._recover_running()

    # Lifecycle Hooks

    def get_stage(self):
        return self.manager.stage

    async def check_workers(self):
        await self.manager.check_workers()

    async def start(self):
        await self.manager.start()

    # RPC methods for Workers

    def restart_workers(self):
        self.manager.request_restart()

    # TODO metric rpc: AtorchEvent, Event
    # TODO diagnosis rpc: NodeFailure, ResourceStats, DiagnosisReportData(XPUTimer)
