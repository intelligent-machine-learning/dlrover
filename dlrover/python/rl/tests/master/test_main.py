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
import time

import ray

from dlrover.python.rl.master.main import DLRoverRLMaster
from dlrover.python.rl.tests.master.base import BaseMasterTest


class RLMasterTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        ray.init()

    def tearDown(self):
        super().tearDown()
        ray.shutdown()

    def test_main(self):
        master_name = "test"
        DLRoverRLMaster.options(name=master_name, lifetime="detached").remote(
            self._job_context.job_config.serialize(),
            self._job_context.rl_context.serialize(),
        )

        # wait master creation
        while True:
            try:
                ray.get_actor(master_name)
                break
            except ValueError:
                time.sleep(1)

        # wait master done
        while True:
            try:
                ray.get_actor(master_name)
                time.sleep(1)
            except ValueError:
                break
