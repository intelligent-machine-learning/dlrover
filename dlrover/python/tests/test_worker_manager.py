# Copyright 2022 The EasyDL Authors. All rights reserved.
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
from datetime import datetime, timedelta

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeStatus,
    NodeType,
    PlatformType,
)
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.worker import ChiefManager, WorkerManager
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client


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
        plan = worker_manager.relaunch_node(
            self._job_nodes[NodeType.WORKER][4]
        )
        self.assertEqual(plan.launch_nodes[0].config_resource.cpu, 16)
        self.assertEqual(worker_manager._nodes[5].id, 5)

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

    def test_cut_pending_node_cpu(self):
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
        plan = worker_manager.cut_pending_node_cpu()
        self.assertEqual(len(plan.launch_nodes), 5)

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
        failed = worker_manager.has_failed_worker()
        self.assertTrue(failed)

        for node in worker_manager._nodes.values():
            node.exit_reason = NodeExitReason.KILLED
        failed = worker_manager.has_failed_worker()
        self.assertFalse(failed)

        wait = worker_manager.wait_worker_restart()
        self.assertTrue(wait)
        for node in worker_manager._nodes.values():
            node.relaunch_count = node.max_relaunch_count

        wait = worker_manager.wait_worker_restart()
        self.assertFalse(wait)
