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

from dlrover.python.common.constants import NodeType, TaskType
from dlrover.python.master.shard.batch_dataset_manager import (
    BatchDatasetManager,
)
from dlrover.python.master.shard.dataset_splitter import (
    PartitionOffsets,
    StreamingDatasetSplitter,
    TableDatasetSplitter,
    TextDatasetSplitter,
)
from dlrover.python.master.shard.streaming_dataset_manager import (
    StreamingDatasetManager,
)


class BatchDatasetTaskMangerTest(unittest.TestCase):
    def test_create_shards(self):
        splitter = TableDatasetSplitter(
            dataset_name="test",
            dataset_size=10000,
            shard_size=100,
            num_epochs=1,
        )
        task_manager = BatchDatasetManager(TaskType.TRAINING, 10, splitter)
        worker_id = 0
        task = task_manager.get_task(NodeType.WORKER, worker_id)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(len(task_manager.todo), 99)
        self.assertEqual(len(task_manager.doing), 1)
        self.assertFalse(task_manager.completed())

        task_manager.report_task_status(task.task_id, True)
        self.assertEqual(len(task_manager.doing), 0)

        for i in range(101):
            task = task_manager.get_task(NodeType.WORKER, worker_id)
            if task.task_id < 0:
                break
            task_manager.report_task_status(task.task_id, True)
        self.assertTrue(task_manager.completed())
        self.assertEqual(task_manager.get_completed_step(), 1000)

    def test_shard_checkpoint_without_shuffle(self):
        splitter = TextDatasetSplitter(
            dataset_name="test",
            dataset_size=10000,
            shard_size=100,
            num_epochs=1,
        )
        ds_manager = BatchDatasetManager(TaskType.TRAINING, 10, splitter)
        worker_id = 0
        task = ds_manager.get_task(NodeType.WORKER, worker_id)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(len(ds_manager.todo), 99)
        self.assertEqual(len(ds_manager.doing), 1)
        self.assertFalse(ds_manager.completed())
        checkpoint = ds_manager.checkpoint()
        self.assertListEqual(checkpoint.todo[0], [100, 200])
        ds_manager.restore_checkpoint(checkpoint)
        self.assertEqual(ds_manager.todo[0].shard.start, 0)
        self.assertEqual(ds_manager.todo[0].shard.end, 100)
        self.assertEqual(ds_manager.todo[0].shard.record_indices, None)

    def test_shard_checkpoint_with_shuffle(self):
        splitter = TextDatasetSplitter(
            dataset_name="test",
            dataset_size=10000,
            shard_size=100,
            num_epochs=1,
            shuffle=True,
        )
        ds_manager = BatchDatasetManager(TaskType.TRAINING, 10, splitter)
        worker_id = 0
        task = ds_manager.get_task(NodeType.WORKER, worker_id)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(len(ds_manager.todo), 99)
        self.assertEqual(len(ds_manager.doing), 1)
        self.assertFalse(ds_manager.completed())
        checkpoint = ds_manager.checkpoint()
        self.assertEqual(len(checkpoint.todo[0][-1]), 100)
        ds_manager.restore_checkpoint(checkpoint)
        self.assertEqual(ds_manager.todo[0].shard.start, 0)
        self.assertEqual(ds_manager.todo[0].shard.end, 100)
        self.assertEqual(len(ds_manager.todo[0].shard.record_indices), 100)


class StreamingDatasetTaskMangerTest(unittest.TestCase):
    def test_create_shards(self):
        partition_offset = PartitionOffsets({0: 1, 1: 0})
        splitter = StreamingDatasetSplitter(
            dataset_name="logstore_test",
            dataset_size=1000,
            shard_size=200,
            partition_offset=partition_offset,
        )
        task_manager = StreamingDatasetManager(TaskType.TRAINING, 10, splitter)
        worker_id = 0
        task = task_manager.get_task(NodeType.WORKER, worker_id)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(len(task_manager.todo), 4)
        self.assertEqual(len(task_manager.doing), 1)
        self.assertFalse(task_manager.completed())
        task_manager.report_task_status(task.task_id, True)
        self.assertEqual(len(task_manager.doing), 0)
        checkpoint = task_manager.checkpoint()
        task_manager.restore_checkpoint(checkpoint)
