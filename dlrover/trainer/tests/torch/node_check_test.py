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

from dlrover.trainer.torch.node_check.ascend_npu import main as npu_main
from dlrover.trainer.torch.node_check.nvidia_gpu import main as gpu_main
from dlrover.trainer.torch.node_check.nvidia_gpu import set_nccl_env
from dlrover.trainer.torch.node_check.utils import mock_error


class TestNetworkCheckScript(unittest.TestCase):
    def setUp(self):
        # Initialization code to run before each test method
        pass

    def tearDown(self):
        # Cleanup code to run after each test method
        pass

    def test_gpu_node_check(self):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        t = gpu_main()
        self.assertTrue(t > 0)
        with open("/tmp/dlrover/network_check/0.txt", "r") as f:
            data = json.load(f)
            self.assertEqual(data["local_rank"], 0)
            self.assertTrue(data["time"] > 0)

        t = npu_main()
        self.assertTrue(t > 0)
        with open("/tmp/dlrover/network_check/0.txt", "r") as f:
            data = json.load(f)
            self.assertEqual(data["local_rank"], 0)
            self.assertTrue(data["time"] > 0)

    def test_mock_error(self):
        raised_error = False
        try:
            mock_error()
        except ValueError:
            raised_error = True
        self.assertFalse(raised_error)

        os.environ["MOCK_ERR_RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        raised_error = False
        try:
            mock_error()
        except ValueError:
            raised_error = True
        self.assertTrue(raised_error)

    def test_set_nccl_env(self):
        set_nccl_env()
        self.assertFalse("NCCL_SOCKET_IFNAME" in os.environ)
        os.environ[
            "NCCL_SETTINGS"
        ] = "NCCL_DEBUG=INFO,NCCL_SOCKET_IFNAME=eth0,NCCL_IB_GID_INDEX=3"
        set_nccl_env()
        self.assertEqual(os.environ["NCCL_SOCKET_IFNAME"], "eth0")
