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


class FileReader:
    def __init__(
        self,
        file_name,
        num_epochs=1,
        batch_size=64,
        enable_dynamic_sharding=True,
        skip_header=True,
    ):

        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._skip_header = skip_header
        self.enable_dynamic_sharding = enable_dynamic_sharding
        self.data_shard_service = None
        self._file_handler = open(file_name, "r")
        self._file_name = file_name

        self.count_data()
        self.data_shard_client = None
        self._consumed_data = 0
        self.build_sharding_client()

    def build_sharding_client(self):
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
                dataset_name=self._file_name,
            )

    def count_data(self):
        self.data = self._file_handler.readlines()
        if self._skip_header:
            self._data_nums = len(self.data) - 1
            self.data = self.data[1:]
        else:
            self._data_nums = len(self.data)

    def iterator(self):
        while True:
            shard = self.data_shard_client.fetch_shard()
            if not shard:
                break
            logger.info("shard is {}".format(shard))
            for i in range(shard.start, shard.end):
                d = self.data[i]
                d = d.strip()
                dd = d.split(",")
                assert len(dd) == 40
                yield d

    def __del__(self):
        if self._file_handler is not None:
            self._file_handler.close()
