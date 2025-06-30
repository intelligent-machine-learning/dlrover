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

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.log import default_logger as logger


def main():
    """
    This entry point is mainly designed for adapting to special scenarios,
    enforcing the use of `dlrover-run`.

    If the dlrover master exists, the `dlrover-run` command is executed
    directly; otherwise, the use of `torchrun` is allowed for execution.
    """

    if env_utils.get_env(NodeEnv.DLROVER_MASTER_ADDR):
        logger.warning(
            "DLRover master exists but using torchrun command. "
            "Replace with dlrover-run directly."
        )
        from dlrover.trainer.torch.elastic_run import main as dlrover_main

        dlrover_main()
    else:
        logger.info(
            "DLRover master not exist so using torchrun command directly."
        )
        from torch.distributed.run import main as torch_main

        torch_main()


if __name__ == "__main__":
    main()
