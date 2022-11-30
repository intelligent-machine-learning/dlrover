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

from dlrover.python.master.shard.dataset_splitter import (
    PartitionOffsets,
    StreamingDatasetSplitter,
    TableDatasetSplitter,
    TextDatasetSplitter,
)


class StreamingDatasetSplitterTest(unittest.TestCase):
    def test_create_streaming_shards_with_dataset_size(self):
        partition_offset = PartitionOffsets({0: 1, 1: 0})
        splitter = StreamingDatasetSplitter(
            dataset_name="logstore_test",
            dataset_size=1000,
            shard_size=200,
            partition_offset=partition_offset,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(shards[0].name, 0)
        self.assertEqual(splitter.epoch, 0)
        shard_0 = shards[0]
        self.assertEqual(shard_0.name, 0)
        self.assertEqual(shard_0.start, 1)
        self.assertEqual(shard_0.end, 201)

    def test_create_streaming_shards_without_dataset_size(self):
        partition_offset = PartitionOffsets({0: 1, 1: 0})
        splitter = StreamingDatasetSplitter(
            dataset_name="test",
            shard_size=200,
            partition_offset=partition_offset,
            fetch_data_size=10000,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(shards[0].name, 0)
        self.assertEqual(splitter.epoch, 0)
        shard_0 = shards[0]
        self.assertEqual(shard_0.name, 0)


class TableDatasetSplitterTest(unittest.TestCase):
    def test_create_shards(self):
        splitter = TableDatasetSplitter(
            dataset_name="test",
            dataset_size=10000,
            shard_size=100,
            num_epochs=1,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(len(shards), 100)
        self.assertEqual(shards[0].start, 0)
        self.assertEqual(shards[0].end, 100)
        self.assertEqual(shards[0].name, "test")
        self.assertEqual(splitter.epoch, 1)

    def test_create_shards_with_huge_dataset(self):
        splitter = TableDatasetSplitter(
            dataset_name="test",
            dataset_size=10000000,
            shard_size=100,
            num_epochs=1,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(len(shards), 50000)
        self.assertEqual(shards[-1].start, 4999900)
        self.assertEqual(shards[-1].end, 5000000)
        self.assertEqual(shards[0].name, "test")
        self.assertEqual(splitter.epoch, 1)
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(shards[0].start, 5000000)
        self.assertEqual(shards[0].end, 5000100)
        self.assertEqual(splitter.epoch, 2)


class TextDatasetSplitterTest(unittest.TestCase):
    def test_create_shards(self):
        splitter = TextDatasetSplitter(
            dataset_name="test",
            dataset_size=1000,
            shard_size=10,
            num_epochs=1,
            shuffle=False,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(len(shards), 100)
        self.assertEqual(shards[0].start, 0)
        self.assertEqual(shards[0].end, 10)
        self.assertListEqual(shards[0].record_indices, list(range(10)))
        self.assertEqual(shards[0].name, "test")
        self.assertEqual(splitter.epoch, 1)

    def test_create_shards_with_shuffle(self):
        splitter = TextDatasetSplitter(
            dataset_name="test",
            dataset_size=1000,
            shard_size=10,
            num_epochs=1,
            shuffle=True,
        )
        splitter.create_shards()
        shards = splitter.get_shards()
        self.assertEqual(len(shards), 100)
        self.assertEqual(shards[0].start, 0)
        self.assertEqual(shards[0].end, 10)
        self.assertNotEqual(shards[0].record_indices, list(range(10)))
        self.assertEqual(shards[0].name, "test")
        self.assertEqual(splitter.epoch, 1)
