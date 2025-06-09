#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shlex
from concurrent.futures import ThreadPoolExecutor

import ray

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import RayMasterClient
from dlrover.python.unified.common.constant import InternalDLConfig
from dlrover.python.unified.trainer.workload import BaseWorkload
from dlrover.trainer.torch.elastic_run import main


@ray.remote
class ElasticWorkload(BaseWorkload):
    """
    Elastic workload is a special workload, it combines both 'trainer' and
    'workload' roles.
    """

    launch_future = None

    def start(self):
        run_cmd = self.config.get(InternalDLConfig.ELASTIC_RUN_CMD)
        env_utils.set_env(NodeEnv.NODE_ID, self.rank)
        env_utils.set_env(NodeEnv.NODE_RANK, self.rank)
        env_utils.set_env(NodeEnv.NODE_TYPE, NodeType.WORKER)

        logger.info(
            f"Run dlrover command in elastic workload: {run_cmd} "
            f"with node-id(rank): {self.rank}"
        )
        launch_future = ThreadPoolExecutor(max_workers=1).submit(
            self._async_launch_agent, run_cmd
        )
        launch_future.add_done_callback(self._handle_launch_future)

    def _async_launch_agent(self, run_cmd):
        try:
            # set master handle's actor id as master address
            RayMasterClient.register_master_actor(self.master_handle)

            main(self.extract_args_from_cmd(run_cmd))
        except Exception as e:
            logger.error(
                "Failed to launch elastic agent for training by "
                f"unexpected error: {e}",
                exc_info=True,
            )
            raise RuntimeError("Agent launch failed")

    def _handle_launch_future(self, future):
        try:
            future.result()
        except Exception as e:
            raise e

    import shlex

    def extract_args_from_cmd(self, run_cmd: str) -> list[str]:
        args_list = shlex.split(run_cmd)

        parsed_args = []
        for arg in args_list[1:]:
            if "=" in arg and arg.startswith("--"):
                key, value = arg.split("=", 1)
                parsed_args.extend([key, value])
            elif arg.startswith("--"):
                parsed_args.append(arg)
            else:
                continue

        return parsed_args
