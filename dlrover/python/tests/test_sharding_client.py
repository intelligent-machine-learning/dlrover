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
    MasterClient,
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
        MasterClient._instance = build_master_client(addr, 0.5)

    def tearDown(self):
        self._master.stop()

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
            num_epochs=1,
            dataset_size=100,
            num_minibatches_per_shard=2,
            dataset_name="test",
        )
        indices = []
        while True:
            try:
                index = client.fetch_sample_index()
                indices.append(index)
            except StopIteration:
                index = None
            if index is None:
                break
        self.assertEqual(len(indices), 100)
        shuffled = False
        for i in range(len(indices)):
            if i != indices[i]:
                print(i, indices[i])
                shuffled = True
                break
        self.assertFalse(shuffled)

    def test_index_sharding_client_with_shuffle(self):
        client = IndexShardingClient(
            batch_size=16,
            num_epochs=1,
            dataset_size=1000,
            num_minibatches_per_shard=2,
            dataset_name="test-0",
            shuffle=True,
        )
        indices = []
        while True:
            try:
                index = client.fetch_sample_index()
                indices.append(index)
            except StopIteration:
                index = None
            if index is None:
                break
        self.assertEqual(len(indices), 1000)
        shuffled = False
        for i in range(len(indices)):
            if i != indices[i]:
                shuffled = True
                break
        self.assertTrue(shuffled)


if __name__ == "__main__":
    unittest.main()
