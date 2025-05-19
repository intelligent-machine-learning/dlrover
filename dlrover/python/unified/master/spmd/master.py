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

import ray

from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.master import BaseMaster


@ray.remote
class SPMDMaster(BaseMaster):
    """
    SPMD master is a control implementation designed for training scenarios
    involving single compute roles.
    """

    def __init__(self, job_config_serialized, dl_context_serialized):
        super().__init__(job_config_serialized, dl_context_serialized)

        self.init()

    def init(self):
        # TODO
        pass

    def _handle_failure(self, failure: FailureDesc):
        pass

    """Remote call functions start"""
    # TODO

    """Remote call functions end"""
