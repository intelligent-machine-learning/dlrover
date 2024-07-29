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

import time
import unittest
from datetime import datetime, timedelta

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeStatus,
    NodeType,
    PlatformType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.worker import ChiefManager, WorkerManager
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client

_dlrover_ctx = Context.singleton_instance()


class WorkerManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        self._job_resource = JobResource()
        self._job_resource.node_group_resources[
            NodeType.WORKER
        ] = NodeGroupResource(5, NodeResource(16, 2048))
        self._elastic_job = new_elastic_job(
            PlatformType.KUBERNETES, "test", "default"
        )
        self._job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

        self._worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

    def test_scale_up_workers(self):
        self._worker_manager._scale_up_workers(3)
        self.assertEqual(len(self._worker_manager._nodes), 8)
        self.assertEqual(self._worker_manager._nodes[7].id, 7)

    def test_scale_down_workers(self):
        workers = list(self._worker_manager._nodes.values())
        self._worker_manager._scale_down_workers(2, workers)
        released_workers = []
        for worker in workers:
            if worker.is_released:
                released_workers.append(worker)
        self.assertEqual(len(released_workers), 2)

    def test_delete_exited_workers(self):
        self._worker_manager._nodes[3].status = NodeStatus.FINISHED
        self._worker_manager._nodes[4].status = NodeStatus.FAILED

        plan = self._worker_manager.delete_exited_workers()
        node_names = [node.name for node in plan.remove_nodes]
        self.assertListEqual(
            node_names,
            ["test-edljob-worker-3", "test-edljob-worker-4"],
        )

    def test_delete_running_workers(self):
        for node in self._worker_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        plan = self._worker_manager.delete_running_workers()
        node_names = [node.name for node in plan.remove_nodes]
        self.assertListEqual(
            node_names,
            [
                "test-edljob-worker-0",
                "test-edljob-worker-1",
                "test-edljob-worker-2",
                "test-edljob-worker-3",
                "test-edljob-worker-4",
            ],
        )

    def test_relaunch_node(self):
        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        failed_worker = self._job_nodes[NodeType.WORKER][4]
        failed_worker.status = NodeStatus.FAILED
        plan = worker_manager.relaunch_node(
            failed_worker, remove_exited_node=True
        )
        self.assertEqual(plan.launch_nodes[0].config_resource.cpu, 16)
        self.assertEqual(worker_manager._nodes[5].id, 5)
        self.assertEqual(plan.remove_nodes[0].config_resource.cpu, 16)

    def test_relaunch_chief_node(self):
        tf_master_node = Node(
            NodeType.MASTER,
            node_id=0,
            config_resource=NodeResource(cpu=16, memory=10240),
        )
        manager = ChiefManager(
            {0: tf_master_node},
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        plan = manager.relaunch_node(tf_master_node)
        self.assertEqual(plan.launch_nodes[0].config_resource.cpu, 16)
        self.assertEqual(manager._nodes[1].id, 1)

    def test_reduce_pending_node_resource(self):
        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in worker_manager._nodes.values():
            node.status = NodeStatus.PENDING
            node.create_time = datetime.now() + timedelta(days=-1)
        plan = worker_manager.reduce_pending_node_resource()
        self.assertEqual(len(plan.launch_nodes), 5)

        for node in worker_manager._nodes.values():
            node.config_resource.gpu_num = 1

        plan = worker_manager.reduce_pending_node_resource()
        self.assertTrue(plan.empty())

    def test_pending_without_workers(self):
        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in worker_manager._nodes.values():
            node.status = NodeStatus.FAILED
            node.exit_reason = NodeExitReason.FATAL_ERROR
        exited = worker_manager.has_exited_worker()
        self.assertTrue(exited)

        for node in worker_manager._nodes.values():
            node.exit_reason = NodeExitReason.KILLED
        exited = worker_manager.has_exited_worker()
        self.assertFalse(exited)

        worker_manager._nodes[0].status = NodeStatus.SUCCEEDED
        exited = worker_manager.has_exited_worker()
        self.assertTrue(exited)

        wait = worker_manager.wait_worker_restart()
        self.assertTrue(wait)
        for node in worker_manager._nodes.values():
            node.relaunch_count = node.max_relaunch_count

        wait = worker_manager.wait_worker_restart()
        self.assertFalse(wait)

    def test_verify_restarting_training(self):
        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        reset = worker_manager.verify_restarting_training(0)
        self.assertFalse(reset)
        worker_manager._nodes[0].restart_training = True
        reset = worker_manager.verify_restarting_training(0)
        self.assertTrue(reset)
        worker_manager._nodes[0].is_released = True
        reset = worker_manager.verify_restarting_training(0)
        self.assertFalse(reset)

    def test_is_training_hang_by_pending(self):
        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self.assertFalse(worker_manager.is_training_hang_by_pending())

        worker_manager.update_node_required_info((4, 8))
        self.assertFalse(worker_manager.is_training_hang_by_pending())

        mock_nodes = {}

        # mock with 3 running + 1 pending short time
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-1)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        self.assertFalse(worker_manager.is_training_hang_by_pending())
        mock_nodes.clear()

        # mock with 3 running + 1 pending long time
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        self.assertTrue(worker_manager.is_training_hang_by_pending())
        mock_nodes.clear()

        # mock with 4 running + 1 pending long time
        for index in range(5):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        self.assertFalse(worker_manager.is_training_hang_by_pending())
        mock_nodes.clear()

        # mock with 3 running + 1 initial long time
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.INITIAL
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        self.assertTrue(worker_manager.is_training_hang_by_pending())

    def test_is_training_hang_by_insufficient_worker(self):
        # mock timeout 2 second(seconds_to_wait_pending_pod * 2)
        _dlrover_ctx.seconds_to_wait_pending_pod = 1

        worker_manager = WorkerManager(
            self._job_nodes[NodeType.WORKER],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self.assertFalse(
            worker_manager.is_training_hang_by_insufficient_worker()
        )

        worker_manager.update_node_required_info((4, 8))
        self.assertFalse(
            worker_manager.is_training_hang_by_insufficient_worker()
        )

        mock_nodes = {}
        is_insufficient = 0

        # mock with 3 running + 1 pending
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        for _ in range(5):
            if worker_manager.is_training_hang_by_insufficient_worker():
                is_insufficient += 1
            time.sleep(1)
        self.assertEqual(is_insufficient, 0)
        mock_nodes.clear()
        is_insufficient = 0

        # mock with 3 running
        for index in range(3):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        for _ in range(5):
            if worker_manager.is_training_hang_by_insufficient_worker():
                is_insufficient += 1
            time.sleep(1)
        self.assertTrue(is_insufficient >= 2)
        mock_nodes.clear()
        is_insufficient = 0

        # mock with 3 running + 1 released
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.DELETED
                mock_node.is_released = True
            mock_nodes[index] = mock_node
        worker_manager._nodes = mock_nodes
        for _ in range(5):
            if worker_manager.is_training_hang_by_insufficient_worker():
                is_insufficient += 1
            time.sleep(1)
        self.assertTrue(is_insufficient >= 2)

        # reset
        _dlrover_ctx.seconds_to_wait_pending_pod = 900
