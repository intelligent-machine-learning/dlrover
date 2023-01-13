# Copyright 2023 The DLRover Authors. All rights reserved.
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
import numpy as np

from dlrover.python.elastic_agent.sharding.client import ShardingClient


def build_data_shard_service(
    batch_size=1,
    num_epochs=1,
    dataset_size=1,
    num_minibatches_per_shard=1,
    dataset_name="iris_training_data",
):
    sharding_client = ShardingClient(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        num_minibatches_per_shard=num_minibatches_per_shard,
    )
    return sharding_client


class FakeReader:
    def __init__(
        self, num_epochs=1, batch_size=64, enable_dynamic_sharding=True
    ):

        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self.enable_dynamic_sharding = enable_dynamic_sharding
        self.data_shard_service = None
        self.count_data()
        self.data_shard_client = None
        self._consumed_data = 0
        self.build_data_shard_client()

    def build_data_shard_client(self):
        if self.enable_dynamic_sharding is True:
            self.data_shard_client = build_data_shard_service(
                batch_size=self._batch_size,
                num_epochs=1,
                dataset_size=self._data_nums,
                num_minibatches_per_shard=1,
                dataset_name="iris_training_data",
            )

    def count_data(self):
        self._data_nums = 10000

    def get_default_shard(self):
        return 1

    def _read_data(self):
        shard = None
        if self.data_shard_client is not None:
            shard = self.data_shard_client.fetch_shard()
        data = self.get_data_by_shard(shard)
        if shard is None:
            data = None
        return data

    def get_data_by_shard(self, shard):
        x = np.random.randint(1, 1000)
        y = 2 * x + np.random.randint(1, 5)
        return "{},{}".format(x, y)

    def iterator(self):
        while True:
            data = self._read_data()
            if data is None:
                break
            yield data
