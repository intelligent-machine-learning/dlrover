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

from abc import ABCMeta, abstractmethod

from dlrover.python.elastic_agent.sharding.client import ShardingClient
from dlrover.trainer.util.log_util import default_logger as logger


def build_sharding_client(
    batch_size=1,
    num_epochs=1,
    dataset_size=1,
    num_minibatches_per_shard=1,
    dataset_name="training_data",
):
    sharding_client = ShardingClient(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        num_minibatches_per_shard=num_minibatches_per_shard,
    )
    return sharding_client


class ElasticReader(metaclass=ABCMeta):
    def __init__(
        self,
        path=None,
    ):
        self._path = path
        self.enable_dynamic_sharding = True
        self.num_minibatches_per_shard = 10

    def set_path(self, path):
        self._path = path

    def set_num_epochs(self, num_epochs):
        self._num_epochs = num_epochs

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_enable_dynamic_sharding(self, enable_dynamic_sharding):
        self._enable_dynamic_sharding = enable_dynamic_sharding

    def set_num_minibatches_per_shard(self, num_minibatches_per_shard):
        self._num_minibatches_per_shard = num_minibatches_per_shard

    def build_sharding_client(self):
        self.count_data()
        if self.enable_dynamic_sharding is True:
            logger.info(
                "Build data shard client in file reader: \n \
                            num_epochs {} \n\
                            batch_size {} \n\
                            data_nums {}".format(
                    self._num_epochs, self._batch_size, self._data_nums
                )
            )
            self.data_shard_client = build_sharding_client(
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                dataset_size=self._data_nums,
                num_minibatches_per_shard=2,
                dataset_name=self._path,
            )

    @abstractmethod
    def read_data_by_index_range(self, start_index, end_index):
        pass

    @abstractmethod
    def count_data(self):
        pass

    def iterator(self):
        while True:
            shard = self.data_shard_client.fetch_shard()
            if not shard:
                break
            logger.info("shard is {}".format(shard))
            for d in self.read_data_by_index_range(shard.start, shard.end):
                yield d
