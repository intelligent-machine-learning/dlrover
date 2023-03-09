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

import json
import unittest

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.constants import NodeType
from dlrover.python.master.shard.task_manager import DatasetShardCheckpoint
from dlrover.python.tests.test_utils import create_task_manager


class TaskMangerTest(unittest.TestCase):
    def test_dispatch_task(self):
        dataset_name = "test"
        task_manager = create_task_manager()
        self.assertEqual(len(task_manager._datasets), 1)
        task = task_manager.get_dataset_task(NodeType.WORKER, 0, dataset_name)
        self.assertEqual(task.task_id, 0)
        dataset_manager = task_manager.get_dataset(dataset_name)
        self.assertIsNotNone(dataset_manager)

        request = elastic_training_pb2.ReportTaskResultRequest()
        request.task_id = 0
        request.dataset_name = dataset_name
        task, worker_id = task_manager.report_dataset_task(request, True)
        self.assertEqual(worker_id, 0)
        self.assertEqual(task.task_id, 0)
        self.assertGreater(task_manager._worker_start_task_time[0], 0)
        self.assertFalse(task_manager.finished())
        task = task_manager.get_dataset_task(NodeType.WORKER, 1, dataset_name)
        epoch = task_manager.get_dataset_epoch(dataset_name)
        self.assertEqual(epoch, 1)

    def test_recover_task(self):
        task_manager = create_task_manager()
        dataset_name = "test"
        dataset = task_manager.get_dataset(dataset_name)
        task = task_manager.get_dataset_task(NodeType.WORKER, 0, dataset_name)
        self.assertEqual(len(dataset.todo), 9)
        request = elastic_training_pb2.ReportTaskResultRequest()
        request.task_id = task.task_id
        request.dataset_name = dataset_name
        task_manager.report_dataset_task(request, False)
        self.assertEqual(len(dataset.todo), 10)
        self.assertEqual(len(dataset.doing), 0)

    def test_dataset_checkpoint(self):
        task_manager = create_task_manager()
        dataset_name = "test"
        task_manager.get_dataset_task(NodeType.WORKER, 0, dataset_name)
        task_manager.get_dataset_task(NodeType.WORKER, 0, dataset_name)
        checkpoint: DatasetShardCheckpoint = (
            task_manager.get_dataset_checkpoint(dataset_name)
        )
        self.assertEqual(checkpoint.dataset_name, dataset_name)
        self.assertListEqual(checkpoint.doing, [[0, 100], [100, 200]])
        self.assertEqual(len(checkpoint.todo), 8)
        self.assertEqual(checkpoint.epoch, 1)
        checkpoint_str = checkpoint.to_json()

        checkpoint_dict = json.loads(checkpoint_str)
        print(checkpoint_dict)
        self.assertDictEqual(
            checkpoint_dict,
            {
                "dataset_name": "test",
                "todo": [
                    [200, 300],
                    [300, 400],
                    [400, 500],
                    [500, 600],
                    [600, 700],
                    [700, 800],
                    [800, 900],
                    [900, 1000],
                ],
                "doing": [[0, 100], [100, 200]],
                "epoch": 1,
                "splitter": None,
            },
        )

        dataset = task_manager.get_dataset(dataset_name)
        task_manager.get_dataset_task(NodeType.WORKER, 0, dataset_name)
        self.assertEqual(len(dataset.todo), 7)
        task_manager.restore_dataset_from_checkpoint(checkpoint_str)
        self.assertEqual(dataset.todo[1].shard.start, 100)
        self.assertEqual(len(dataset.todo), 10)

    def test_task_hang(self):
        task_manager = create_task_manager()
        dataset_name = "test"
        dataset = task_manager.get_dataset(dataset_name)
        dataset._latest_task_end_time = 3600
        hang = task_manager.task_hanged()
        self.assertTrue(hang)
