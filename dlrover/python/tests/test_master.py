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

import unittest

from dlrover.python.common.constants import JobExitReason, NodeStatus, NodeType
from dlrover.python.master.master import Master
from dlrover.python.master.shard.dataset_splitter import new_dataset_splitter
from dlrover.python.tests.test_utils import MockK8sJobArgs


class MasterTest(unittest.TestCase):
    def setUp(self) -> None:
        params = MockK8sJobArgs()
        params.initilize()
        self.master = Master(2222, params)

    def test_exit_by_workers(self):
        self.master.job_manager._init_nodes()
        job_nodes = self.master.job_manager._job_nodes
        for node in job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.CHIEF].values():
            node.status = NodeStatus.FINISHED
        self.master.run()
        self.assertEqual(self.master._exit_code, 0)
        self.assertEqual(self.master._exit_reason, JobExitReason.SUCCEEDED)

    def test_exit_by_tasks(self):
        self.master.job_manager._init_nodes()
        job_nodes = self.master.job_manager._job_nodes
        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.CHIEF].values():
            node.status = NodeStatus.FINISHED

        job_nodes[NodeType.WORKER][0].status = NodeStatus.FINISHED

        splitter = new_dataset_splitter(
            False,
            100,
            10000,
            1,
            "test",
            "table",
        )

        self.master.task_manager.new_dataset(10, 10000, "test", splitter)

        for dataset in self.master.task_manager._datasets.values():
            dataset.todo.clear()
            dataset.doing.clear()
            dataset._dataset_splitter.epoch = 10
        self.master.run()
        self.assertEqual(self.master._exit_code, 0)
        self.assertEqual(self.master._exit_reason, JobExitReason.SUCCEEDED)
