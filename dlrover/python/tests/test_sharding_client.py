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

from dlrover.python.elastic_agent.master_client import (
    GlobalMasterClient,
    LocalDataset,
    build_master_client,
)
from dlrover.python.elastic_agent.sharding.client import (
    IndexShardingClient,
    ShardingClient,
)
from dlrover.python.tests.test_utils import start_local_master


class DataShardClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        GlobalMasterClient.MASTER_CLIENT = build_master_client(addr)

    def addCleanup(self):
        self._master.stop()

    def test_local_dataset(self):
        dataset = LocalDataset(
            batch_size=16,
            num_epochs=2,
            dataset_size=100,
            shuffle=False,
            num_minibatches_per_shard=2,
        )
        dataset.create_tasks()
        self.assertEqual(len(dataset._todo), 4)
        start, end = dataset.get_task()
        self.assertEqual(start, 0)
        self.assertEqual(end, 32)

    def test_sharding_client(self):
        data_shard_service = ShardingClient(
            batch_size=16,
            num_epochs=2,
            dataset_size=100,
            num_minibatches_per_shard=2,
            dataset_name="test",
        )
        shard = data_shard_service.fetch_shard()
        self.assertEqual(shard.start, 0)
        self.assertEqual(shard.end, 32)
        shard_count = 1
        while True:
            shard = data_shard_service.fetch_shard()
            if not shard:
                break
            data_shard_service.report_batch_done(32)
            shard_count += 1
        self.assertEqual(shard_count, 8)
        checkpoint_str = data_shard_service.get_shard_checkpoint()
        checkpoint = json.loads(checkpoint_str)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(len(checkpoint["todo"]), 0)
        data_shard_service.restore_shard_from_checkpoint(checkpoint_str)

    def test_index_sharding_client(self):
        client = IndexShardingClient(
            batch_size=16,
            num_epochs=2,
            dataset_size=100,
            num_minibatches_per_shard=2,
            dataset_name="test",
        )
        index = client.fetch_sample_index()
        self.assertEqual(index, 0)
        sample_count = 1
        while True:
            index = client.fetch_sample_index()
            if not index:
                break
            sample_count += 1
        self.assertEqual(sample_count, 100)


if __name__ == "__main__":
    unittest.main()
