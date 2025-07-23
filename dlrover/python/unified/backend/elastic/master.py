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


from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.unified.controller.api import PrimeMasterApi

from .manager import ElasticManager


class ElasticMaster(ActorBase):
    def _setup(self):
        workers = PrimeMasterApi.get_workers_by_role(self.actor_info.role)
        self.manager = ElasticManager(workers)

        self.manager._prepare()
        self._update_stage_force(WorkerStage.READY)

    # Lifecycle Hooks

    def status(self):
        if self.manager.finished:
            self._update_stage_force(WorkerStage.FINISHED)
        return super().status()

    async def check_workers(self):
        await self.manager.check_workers()

    async def start(self):
        assert self.stage == WorkerStage.READY, (
            f"Cannot start ElasticMaster in stage {self.stage}, expected READY."
        )
        await self.manager.start()
        self._update_stage_force(WorkerStage.RUNNING)

    # RPC methods for Workers

    # TODO metric rpc: AtorchEvent, Event
    # TODO diagnosis rpc: NodeFailure, ResourceStats, DiagnosisReportData(XPUTimer)
