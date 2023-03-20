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

from torch.utils.data import Dataset

from dlrover.python.elastic_agent.sharding.client import IndexShardingClient


def read_txt(path):
    with open(path, "r") as fp:
        content = fp.readlines()
        return content


class ElasticDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, path, batch_size, epochs, shuffle):
        """Using ElasticDataset, the node can read samples without
        duplicates with other nodes in an epoch. DLRover master
        will dispatch the index of sample in a dataset to one node.

        Args:
            path: str, the path of dataset meta file. For example, if the image
                is stored in a folder. The meta file should be a
                text file where each line is the absolute path of a image.
            batch_size: int, the size of batch samples to compute gradients
                in a trainer process.
            epochs: int, the number of epoch.
            shuffle: bool, whether to shuffle samples in the dataset.
        """
        self.lines = read_txt(path)
        self.dataset_size = len(self.lines)
        self._shard_client = IndexShardingClient(
            dataset_name=path,
            batch_size=batch_size,
            num_epochs=epochs,
            dataset_size=self.dataset_size,
            shuffle=shuffle,
            storage_type="text",
        )

    def __len__(self):
        return self._shard_client.get_total_sample_num()

    def __getitem__(self, _):
        index = self._shard_client.fetch_sample_index()
        return self.read_sample(index)

    def get_epoch(self):
        self._shard_client.get_current_epoch()

    def report_batch_done(self, batch_size=None):
        """After updating models using the samples, the dataset need to
        report the batch completion."""
        self._shard_client.report_batch_done(batch_size)

    def end_shard(self):
        return self._shard_client.no_more_data

    @abstractmethod
    def read_sample(self, index):
        """Implement to read sample data by the index."""
        pass
