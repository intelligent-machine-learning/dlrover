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

from dlrover.python.common.constants import EngineType, NodeStatus, NodeType
from dlrover.python.master.node.worker import WorkerManager
from dlrover.python.master.resource.job import JobResourceConfig
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client


class WorkerManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        self._job_resource = JobResourceConfig()
        self._job_resource.add_node_group_resource(
            NodeType.WORKER, 5, "cpu=16,memory=2048Mi", ""
        )
        self._elastic_job = new_elastic_job(
            EngineType.KUBERNETES, "test", "default"
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
        plan = self._worker_manager.scale_up_workers(3)
        self.assertEqual(plan.node_group_resources[NodeType.WORKER].count, 8)
        self.assertEqual(len(self._worker_manager._nodes), 8)
        self.assertEqual(self._worker_manager._nodes[7].id, 7)

    def test_scale_down_workers(self):
        workers = list(self._worker_manager._nodes.values())
        plan = self._worker_manager.scale_down_workers(2, workers)
        self.assertListEqual(
            plan.removed_nodes,
            ["test-edljob-worker-4", "test-edljob-worker-3"],
        )

    def test_delete_exited_workers(self):
        self._worker_manager._nodes[3].status = NodeStatus.FINISHED
        self._worker_manager._nodes[4].status = NodeStatus.FAILED

        plan = self._worker_manager.delete_exited_workers()
        self.assertListEqual(
            plan.removed_nodes,
            ["test-edljob-worker-3", "test-edljob-worker-4"],
        )

    def test_delete_running_workers(self):
        for node in self._worker_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        plan = self._worker_manager.delete_running_workers()
        self.assertListEqual(
            plan.removed_nodes,
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
        self.assertEqual(plan.node_resources["test-edljob-worker-4"].cpu, 16)
        self.assertEqual(worker_manager._nodes[5].id, 5)

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
        self.assertEqual(len(plan.node_resources), 5)
