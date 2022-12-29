# Copyright 2022 The DLRover Authors. All rights reserved.
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

import os

from dlrover.trainer.platform import starter
from dlrover.trainer.util.log_util import default_logger as logger

if __name__ == "__main__":
    logger.info(
        "WORKFLOW_ID: %s, USERNUMBER: %s",
        os.environ.get("WORKFLOW_ID", None),
        os.environ.get("USERNUMBER", None),
    )

    logger.info("local entry is running")
    starter.run()
