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

import asyncio

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.unified.contoller.api import PrimeMasterApi

from .manager import ElasticManager
from .servicer import RayMasterServicer


class ElasticMaster(ActorBase):
    def _setup(self):
        nodes = PrimeMasterApi.get_workers_by_role(self.node_info.role)
        self.manager = ElasticManager(nodes)

        self.manager._prepare()
        self._init_service()

    # Lifecycle Hooks

    def status(self):
        if self.manager.finished:
            self._update_stage_force(WorkerStage.FINISHED)
        return super().status()

    def self_check(self):
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return
        logger.info("Elastic Master self check")

    async def check_workers(self):
        await self.manager.check_workers()

    async def start(self):
        if not self._update_stage_if(WorkerStage.RUNNING, WorkerStage.PENDING):
            return
        await self.manager.start()

    # RPC methods for Workers

    # TODO(longtime): merge Servicer, flatten use Ray rpc.

    def _init_service(self):
        self._service_handler = RayMasterServicer(self.manager)

    async def agent_report(self, request):
        logger.debug(f"Got agent report call: {request}")
        response = await asyncio.to_thread(
            self._service_handler.agent_report, request
        )
        logger.debug(f"Response agent report call: {response}")
        return response

    async def agent_get(self, request):
        logger.debug(f"Got agent get call: {request}")
        response = await asyncio.to_thread(
            self._service_handler.agent_get, request
        )
        logger.debug(f"Response agent get call: {response}")
        return response
