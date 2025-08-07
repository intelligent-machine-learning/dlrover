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
import time
import unittest

from dlrover.python.common.comm import ParallelConfig
from dlrover.python.common.constants import (
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.node import Node, NodeEvent, NodeResource


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_is_unrecoverable_failure(self):
        node = Node("worker", 0)
        self.assertEqual(node.get_unrecoverable_failure_msg(), "unknown")
        node.critical = True
        self.assertTrue("critical" in node.get_unrecoverable_failure_msg())

        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.relaunch_count = 3
        node.config_resource.gpu_num = 1
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertTrue("exhausted" in node.get_unrecoverable_failure_msg())

        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.config_resource.gpu_num = 1
        node.exit_reason = NodeExitReason.FATAL_ERROR
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertTrue("fatal" in node.get_unrecoverable_failure_msg())

        node = Node("worker", 0)
        node.max_relaunch_count = 3
        node.config_resource.gpu_num = 0
        node.config_resource.memory = NodeResourceLimit.MAX_MEMORY
        node.exit_reason = NodeExitReason.OOM
        is_unrecoverable = node.is_unrecoverable_failure()
        self.assertEqual(is_unrecoverable, True)
        self.assertTrue("oom" in node.get_unrecoverable_failure_msg())

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
        self.assertEqual(
            node.get_reported_status(), NodeEventType.SUCCEEDED_EXITED
        )

        node.update_reported_status(NodeEventType.NODE_CHECK_FAILED)
        self.assertTrue(node.is_succeeded_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.update_reported_status(NodeEventType.FAILED_EXITED)
        self.assertTrue(node.is_succeeded_and_exited())
        self.assertFalse(node.is_failed_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.reported_status = (NodeEventType.FAILED_EXITED, 0)
        self.assertFalse(node.is_succeeded_and_exited())
        self.assertTrue(node.is_failed_and_exited())
        self.assertTrue(node.is_exited_reported())

        node.update_from_node(node)
        node.id = 100
        node.update_from_node(node)

        node = node.generate_relaunch_node(123)
        self.assertEqual(node.id, 123)
        self.assertFalse(node.reported_status[0])

    def test_node_event(self):
        node = Node("worker", 0)
        event = NodeEvent(NodeEventType.NODE_CHECK_FAILED, node)
        self.assertTrue(event.is_node_check_event())
        self.assertFalse(event.is_pre_check_event())

        event = NodeEvent(NodeEventType.WAIT_PRE_CHECK, node)
        self.assertFalse(event.is_node_check_event())
        self.assertTrue(event.is_pre_check_event())

    def test_node_status(self):
        self.assertFalse(NodeStatus.is_terminal_status(NodeStatus.INITIAL))
        self.assertFalse(NodeStatus.is_terminal_status(NodeStatus.RUNNING))
        self.assertTrue(NodeStatus.is_terminal_status(NodeStatus.DELETED))
        self.assertTrue(NodeStatus.is_terminal_status(NodeStatus.FAILED))

    def test_get_name(self):
        node = Node("worker", 5)
        name = node.get_name()
        self.assertEqual(name, "5")

        node.name = "node"
        name = node.get_name()
        self.assertEqual(name, "node")

    def test_group(self):
        node = Node("worker", 0)
        self.assertFalse(node.has_group())

        node = Node("worker", 0, node_group=0, node_group_size=1)
        self.assertTrue(node.has_group())
        node = Node("worker", 0, node_group=0, node_group_size=None)
        self.assertFalse(node.has_group())
        node = Node("worker", 0, node_group=None, node_group_size=None)
        self.assertFalse(node.has_group())

    def test_generate_relaunch_node(self):
        node = Node(
            NodeType.WORKER,
            1,
            NodeResource(2, 2000),
            name="test-1",
            status=NodeStatus.RUNNING,
            start_time=time.time(),
            rank_index=1,
            relaunch_count=2,
            critical=False,
            max_relaunch_count=5,
            relaunchable=True,
            service_addr="test-1.service",
            host_name="test-1.host",
            host_ip="1.1.1.1",
            paral_config=ParallelConfig(),
            restart_training=True,
            node_group=1,
            node_group_size=8,
            node_group_id=3,
        )
        node.init_time = time.time()
        node.start_hang_time = time.time()
        node.finish_time = time.time()
        node.heartbeat_time = time.time()
        node.hang = True
        node.is_released = True

        relaunch_node = node.generate_relaunch_node(2, "test-2")
        self.assertNotEqual(node.id, relaunch_node.id)
        self.assertNotEqual(node.name, relaunch_node.name)
        self.assertNotEqual(
            node.config_resource, relaunch_node.config_resource
        )
        self.assertNotEqual(node.init_time, relaunch_node.init_time)
        self.assertNotEqual(node.start_time, relaunch_node.start_time)
        self.assertNotEqual(node.finish_time, relaunch_node.finish_time)
        self.assertNotEqual(node.heartbeat_time, relaunch_node.heartbeat_time)
        self.assertNotEqual(
            node.start_hang_time, relaunch_node.start_hang_time
        )
        self.assertNotEqual(node.status, relaunch_node.status)
        self.assertNotEqual(node.is_released, relaunch_node.is_released)
        self.assertNotEqual(node.host_ip, relaunch_node.host_ip)
        self.assertNotEqual(node.host_name, relaunch_node.host_name)

        self.assertEqual(
            node.config_resource.to_resource_dict(),
            relaunch_node.config_resource.to_resource_dict(),
        )
        self.assertEqual(node.type, relaunch_node.type)
        self.assertEqual(node.rank_index, relaunch_node.rank_index)
        self.assertEqual(node.relaunch_count, relaunch_node.relaunch_count)
        self.assertEqual(node.critical, relaunch_node.critical)
        self.assertEqual(
            node.max_relaunch_count, relaunch_node.max_relaunch_count
        )
        self.assertEqual(node.relaunchable, relaunch_node.relaunchable)
        self.assertEqual(node.service_addr, relaunch_node.service_addr)
        self.assertEqual(node.paral_config, relaunch_node.paral_config)
        self.assertEqual(node.group, relaunch_node.group)
        self.assertEqual(node.group_id, relaunch_node.group_id)
        self.assertEqual(node.group_size, relaunch_node.group_size)
