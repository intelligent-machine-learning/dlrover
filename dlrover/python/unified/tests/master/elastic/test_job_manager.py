#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import time
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import (
    JobStage,
    NodeEventType,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.node import Node, NodeEvent
from dlrover.python.unified.master.elastic.job_manager import ElasticJobManager
from dlrover.python.unified.tests.master.elastic.base import ElasticBaseTest


class ElasticJobManagerTest(ElasticBaseTest):
    def setUp(self):
        super().setUp()
        patcher = patch("asyncio.get_event_loop")
        self.addCleanup(patcher.stop)
        self.mock_get_loop = patcher.start()
        self.mock_get_loop.return_value = MagicMock()

    def test_basic(self):
        job_manager = ElasticJobManager()
        self.assertIsNotNone(job_manager.elastic_context)
        self.assertIsNotNone(job_manager.get_executor())

        job_manager.create_workloads = MagicMock(return_value=None)
        job_manager.setup_workloads = MagicMock(return_value=None)
        job_manager.execute = MagicMock(return_value=None)

        job_manager.start_job()
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("node_monitor", active_threads_name)

        self.assertFalse(job_manager.has_job_error())

        job_manager.stop_job()

    def test_update_node_paral_config(self):
        job_manager = ElasticJobManager()
        job_manager.update_node_paral_config(NodeType.WORKER, 0, "test")

        job_manager._elastic_context._job_nodes = {
            NodeType.WORKER: {0: Node(node_type=NodeType.WORKER, node_id=0)}
        }
        job_manager.update_node_paral_config(NodeType.WORKER, 0, "test")
        self.assertEqual(
            job_manager._elastic_context._job_nodes[NodeType.WORKER][
                0
            ].paral_config,
            "test",
        )

    def test_collect_node_heartbeat(self):
        job_manager = ElasticJobManager()
        ts = time.time()
        job_manager.collect_node_heart_beat(NodeType.WORKER, 0, ts)

        job_manager._elastic_context._job_nodes = {
            NodeType.WORKER: {0: Node(node_type=NodeType.WORKER, node_id=0)}
        }
        job_manager.collect_node_heart_beat(NodeType.WORKER, 0, ts)
        self.assertEqual(
            job_manager._elastic_context.job_node(
                NodeType.WORKER, 0
            ).heartbeat_time,
            ts,
        )

    def test_get_running_nodes(self):
        job_manager = ElasticJobManager()
        self.assertFalse(job_manager.get_running_nodes())

        job_manager._elastic_context._job_nodes = {
            NodeType.WORKER: {
                0: Node(
                    node_type=NodeType.WORKER,
                    node_id=0,
                    status=NodeStatus.RUNNING,
                )
            }
        }
        self.assertEqual(len(job_manager.get_running_nodes()), 1)

    def test_update_resource_usage(self):
        job_manager = ElasticJobManager()
        job_manager.update_node_resource_usage(NodeType.WORKER, 0, 0.1, 128)

        node0 = Node(node_type=NodeType.WORKER, node_id=0)
        job_manager._elastic_context._job_nodes = {NodeType.WORKER: {0: node0}}
        job_manager.update_node_resource_usage(NodeType.WORKER, 0, 0.1, 128)
        self.assertEqual(node0.used_resource.cpu, 0.1)
        self.assertEqual(node0.used_resource.memory, 128)
        self.assertEqual(node0.start_hang_time, 0)

        node0.config_resource.cpu = 10
        job_manager.update_node_resource_usage(NodeType.WORKER, 0, 0.1, 128)
        self.assertNotEquals(node0.start_hang_time, 0)

        job_manager.update_node_resource_usage(NodeType.WORKER, 0, 5, 128)
        self.assertEqual(node0.start_hang_time, 0)

    def test_process_reported_node_event(self):
        job_manager = ElasticJobManager()
        node0 = Node(node_type=NodeType.WORKER, node_id=0)
        node_event = NodeEvent(NodeEventType.SUCCEEDED_EXITED, node0)

        job_manager.process_reported_node_event(node_event)
        self.assertEqual(
            job_manager.elastic_context.get_job_stage(), JobStage.JOB_STOPPING
        )
