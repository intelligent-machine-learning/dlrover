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
import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import InternalDLConfig
from dlrover.python.unified.trainer.workload import BaseWorkload
from dlrover.trainer.torch.elastic_run import main


@ray.remote
class ElasticWorkload(BaseWorkload):
    def start(self):
        run_cmd = self.config.get(InternalDLConfig.ELASTIC_RUN_CMD)
        logger.info(f"Run dlrover command in elastic workload: {run_cmd}")

        run_cmd_args = run_cmd.split("dlrover-run")[1].strip()
        main(run_cmd_args)
