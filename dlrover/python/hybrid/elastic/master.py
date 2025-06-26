from dlrover.python.common.log import default_logger as logger
from dlrover.python.hybrid.common.node_defines import ActorBase
from dlrover.python.hybrid.elastic.manager import ElasticManager
from dlrover.python.hybrid.elastic.servicer import RayMasterServicer


class ElasticMaster(ActorBase):
    def __init__(self, master_config):
        self.manager = ElasticManager(master_config)

        self._init_service()

    def status(self):
        print("Elastic Master is running")

    def self_check(self):
        pass

    def check_workers(self):
        pass

    def start(self):
        self.manager.start()

    def stop(self):
        self.manager.stop()

    ## RPC methods

    def _init_service(self):
        self._service_handler = RayMasterServicer(self.manager)

    async def agent_report(self, request):
        logger.debug(f"Got agent report call: {request}")
        response = self._service_handler.agent_report(request)
        logger.debug(f"Response agent report call: {response}")
        return response

    async def agent_get(self, request):
        logger.debug(f"Got agent get call: {request}")
        response = self._service_handler.agent_get(request)
        logger.debug(f"Response agent get call: {response}")
        return response
