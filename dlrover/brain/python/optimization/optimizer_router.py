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

from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
)


class OptimizerRouter:
    """
    OptimizerRouter routes optimize requests to the best optimizers
    based runtime status of this job.
    """

    def __init__(self):
        pass

    def route(self, job: JobMeta, conf: OptimizeConfig) -> str:
        # todo: pick up optimizer based on job information
        return conf.optimizer_name
