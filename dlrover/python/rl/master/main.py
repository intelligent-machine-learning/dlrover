# Copyright 2025 The EasyDL Authors. All rights reserved.
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

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.context import RLContext, get_job_context


@ray.remote
class DLRoverRLMaster(object):
    def __init__(self, job_config_serialized, rl_context_serialized):
        self._job_config = JobConfig.deserialize(job_config_serialized)
        self._rl_context = RLContext.deserialize(rl_context_serialized)
        self._job_context = get_job_context()
        self._job_context.init(self._job_config, self._rl_context)

        self._started = False

        logger.infof(
            "DLRover RLMaster initiated with "
            f"job-config: {self._job_config}, "
            f"rl-context: {self._rl_context}."
        )
        self.start()

    def start(self):
        self._started = True

        # load context from backend
