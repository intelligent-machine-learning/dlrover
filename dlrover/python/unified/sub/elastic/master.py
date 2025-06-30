import asyncio

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.common.node_defines import ActorBase, WorkerStage
from dlrover.python.unified.prime.api import PrimeMasterApi

from .manager import ElasticManager
from .servicer import RayMasterServicer


class ElasticMaster(ActorBase):
    def _setup(self):
        nodes = PrimeMasterApi.get_workers_by_role(self.node_info.role)
        self.manager = ElasticManager(nodes)

        self.manager._prepare()
        self._init_service()

    def status(self):
        if JobStage.is_ending_stage(self.manager.stage):
            self._update_stage_force(WorkerStage.FINISHED)
        return super().status()

    def self_check(self):
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return
        logger.info("Elastic Master self check")

    def check_workers(self):
        pass

    async def start(self):
        if not self._update_stage_if(WorkerStage.RUNNING, WorkerStage.PENDING):
            return
        await self.manager.start()

    ## RPC methods

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
