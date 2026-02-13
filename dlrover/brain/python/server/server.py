from fastapi import FastAPI, status
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse, JobConfigResponse, JobConfigRequest,
)
import uvicorn

from dlrover.brain.python.jobmanagement.job_config import JobConfigManager
from dlrover.brain.python.optimization.optimizer_manager import OptimizerManager
from dlrover.brain.python.common.args import get_parsed_args


class BrainServer:
    def __init__(self):
        self._server = FastAPI(title="DLRover Brain Server")
        self._optimizer_manager = OptimizerManager()
        self._config_manager = JobConfigManager()

    def register_routes(self):
        # We map the URL to the class method 'self.optimize'
        self._server.get(
            "/optimize",
            response_model=OptimizeResponse,
            status_code=status.HTTP_201_CREATED
        )(self.optimize)

        self._server.get(
            "/job_config",
            response_model=JobConfigResponse,
             status_code=status.HTTP_201_CREATED
        )(self.job_config)

    def optimize(self, request: OptimizeRequest):
        plan = self._optimizer_manager.optimize(request.job, request.config)
        return OptimizeResponse(
            job_opt_plan=plan,
        )

    def job_config(self, request: JobConfigRequest):
        config = self._config_manager.get_job_config(request.job)
        return JobConfigResponse(
            job_configs=config.convert_to_dict(),
        )

    def run(self, args):
        host = "0.0.0.0"
        port = args.port
        uvicorn.run(self._server, host=host, port=port)


if __name__ == "__main__":
    args = get_parsed_args()
    brain_server = BrainServer()
    brain_server.run(args)
