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
import os
import unittest

import ray

from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.driver.main import main
from dlrover.python.rl.tests.test_data import TestData


class DriverTest(unittest.TestCase):
    def setUp(self):
        os.environ[RLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
        ray.init()

    def tearDown(self):
        os.environ.clear()
        ray.shutdown()

    def test_driver(self):
        args = [
            "--job_name",
            "test",
            "--master_cpu",
            "1",
            "--master_memory",
            "100",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]

        main(args)
        self.assertIsNotNone(ray.get_actor("DLRoverRLMaster-test"))
