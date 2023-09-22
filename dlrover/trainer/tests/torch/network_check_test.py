# Copyright 2023 The DLRover Authors. All rights reserved.
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

import json
import os
import unittest

from dlrover.trainer.torch.run_network_check import (
    FAULT_CHECK_TASK,
    main,
    matmul,
)


class TestNetworkCheckScript(unittest.TestCase):
    def setUp(self):
        # Initialization code to run before each test method
        pass

    def tearDown(self):
        # Cleanup code to run after each test method
        pass

    def test_fault_check(self):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        t = main(FAULT_CHECK_TASK)
        self.assertTrue(t > 0)
        with open("/tmp/dlrover/network_check/0.txt", "r") as f:
            data = json.load(f)
            self.assertEqual(data["local_rank"], 0)
            self.assertTrue(data["time"] > 0)

    def test_matmul(self):
        t = matmul(False, 0)
        self.assertTrue(t > 0)
