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
import unittest

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv


class EnvUtilsTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(NodeEnv.NODE_ID, None)
        os.environ.pop(NodeEnv.NODE_RANK, None)
        os.environ.pop(NodeEnv.NODE_NUM, None)
        os.environ.pop(NodeEnv.WORKER_RANK, None)
        os.environ.pop("LOCAL_WORLD_SIZE", None)

    def test_get_env(self):
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 0)

        os.environ[NodeEnv.WORKER_RANK] = "1"
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 1)

        os.environ[NodeEnv.NODE_RANK] = "2"
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 2)

        os.environ[NodeEnv.NODE_NUM] = "4"
        node_num = env_utils.get_node_num()
        self.assertEqual(node_num, 4)

        os.environ[NodeEnv.NODE_ID] = "1"
        node_id = env_utils.get_node_id()
        self.assertEqual(node_id, 1)

        node_type = env_utils.get_node_type()
        self.assertEqual(node_type, "worker")

        os.environ["LOCAL_WORLD_SIZE"] = "8"
        size = env_utils.get_local_world_size()
        self.assertEqual(size, 8)
