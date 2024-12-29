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

from dlrover.python.common.constants import (
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
)
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
        self.assertFalse(node.is_resource_scalable())
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
        self.assertTrue(node.is_resource_scalable())
        self.assertEqual(is_unrecoverable, True)
        self.assertEqual("oom" in node.unrecoverable_failure_msg, True)

        node.update_reported_status(NodeEventType.NODE_CHECK_SUCCEEDED)
        self.assertFalse(node.is_succeeded_and_exited())
        self.assertFalse(node.is_exited_reported())
        self.assertFalse(node.is_node_check_failed())
        node.update_reported_status(NodeEventType.NODE_CHECK_FAILED)
        self.assertFalse(node.is_succeeded_and_exited())
        self.assertFalse(node.is_exited_reported())
        self.assertTrue(node.is_node_check_failed())

        self.assertFalse(node.is_succeeded_and_exited())
        node.update_reported_status(NodeEventType.SUCCEEDED_EXITED)
        self.assertTrue(node.is_succeeded_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.update_reported_status(NodeEventType.NODE_CHECK_FAILED)
        self.assertTrue(node.is_succeeded_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.update_reported_status(NodeEventType.FAILED_EXITED)
        self.assertTrue(node.is_succeeded_and_exited())
        self.assertFalse(node.is_failed_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.reported_status = NodeEventType.FAILED_EXITED
        self.assertFalse(node.is_succeeded_and_exited())
        self.assertTrue(node.is_failed_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.update_from_node(node)
        node.id = 100
        node.update_from_node(node)

        node = node.get_relaunch_node_info(123)
        self.assertEqual(node.id, 123)
        self.assertFalse(node.reported_status)
