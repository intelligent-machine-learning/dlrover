from fastapi import FastAPI, status
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
)
import uvicorn
from dlrover.brain.python.optimizer.optimizer_manager import OptimizerManager
from dlrover.brain.python.common.args import get_parsed_args


class BrainServer:
    def __init__(self):
        self.server = FastAPI(title="DLRover Brain Server")
        self.optimizer_manager = OptimizerManager()

    def register_routes(self):
        # We map the URL to the class method 'self.optimize'
        self.server.get(
            "/optimize",
            response_model=OptimizeResponse,
            status_code=status.HTTP_201_CREATED
        )(self.optimize)

    def optimize(self, request: OptimizeRequest):
        plan = self.optimizer_manager.optimize(request.job, request.config)
        return OptimizeResponse(
            job_opt_plan=plan,
        )

    def run(self, args):
        host = "0.0.0.0"
        port = args.port
        uvicorn.run(self.server, host=host, port=port)


if __name__ == "__main__":
    args = get_parsed_args()
    brain_server = BrainServer()
    brain_server.run(args)
