# Copyright 2026 The DLRover Authors. All rights reserved.
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

from fastapi import FastAPI, status
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
    JobConfigResponse,
    JobConfigRequest,
)
import uvicorn

from dlrover.brain.python.jobmanagement.job_config_manager import (
    JobConfigManager,
)
from dlrover.brain.python.optimization.optimizer_manager import (
    OptimizerManager,
)
from dlrover.brain.python.common.args import get_parsed_args


class BrainServer:
    """
    The training jobs access the service of DLRover brain via BrainServer.
    The BrainServer provides two APIs via HTTP protocol:
        - optimize: optimize the resource configuration of a training job.
        - get_config: retrieve the configuration of a job.
    """

    def __init__(self):
        self._server = FastAPI(title="DLRover Brain Server")
        self._optimizer_manager = OptimizerManager()
        self._config_manager = JobConfigManager()

        self.register_routes()

    def register_routes(self):
        # We map the URL to the class method 'self.optimize'
        self._server.post(
            "/optimize",
            response_model=OptimizeResponse,
            status_code=status.HTTP_201_CREATED,
        )(self.optimize)

        self._server.post(
            "/job_config",
            response_model=JobConfigResponse,
            status_code=status.HTTP_200_OK,
        )(self.job_config)

    def optimize(self, request: OptimizeRequest):
        plan = self._optimizer_manager.optimize(request.job, request.config)
        return OptimizeResponse(
            job_opt_plan=plan,
        )

    def job_config(self, request: JobConfigRequest):
        config = self._config_manager.get_job_config(request.job)
        if config is None:
            vals = {}
        else:
            vals = config.get_config_values()
        return JobConfigResponse(
            job_configs=vals,
        )

    def run(self, args):
        host = "0.0.0.0"
        port = args.port
        uvicorn.run(self._server, host=host, port=port)


if __name__ == "__main__":
    args = get_parsed_args()
    brain_server = BrainServer()
    brain_server.run(args)
