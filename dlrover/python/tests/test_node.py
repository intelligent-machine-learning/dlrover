# Copyright 2024 The DLRover Authors. All rights reserved.
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

import unittest

from dlrover.python.common.constants import NodeExitReason, NodeResourceLimit
from dlrover.python.common.node import Node


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_is_unrecoverable_failure(self):
        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.relaunch_count = 3
        node.config_resource.gpu_num = 1
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertEqual("exhausted" in node.unrecoverable_failure_msg, True)

        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.config_resource.gpu_num = 1
        node.exit_reason = NodeExitReason.FATAL_ERROR
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertEqual("fatal" in node.unrecoverable_failure_msg, True)

        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.config_resource.gpu_num = 0
        node.config_resource.memory = NodeResourceLimit.MAX_MEMORY
        node.exit_reason = NodeExitReason.OOM
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertEqual("oom" in node.unrecoverable_failure_msg, True)
