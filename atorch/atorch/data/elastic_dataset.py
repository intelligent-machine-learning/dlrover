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


class ElasticDataset(Dataset, metaclass=ABCMeta):

    """Using ElasticDataset, the node can read samples without
    duplicates with other nodes in an epoch. DLRover master
    will dispatch the index of sample in a dataset to one node.
    Users need to implement the read_sample to read data by the
    sample index.

    Example:
    >>> class MyElasticDataset(ElasticDataset):
    >>>     def __init__(self):
    >>>         self.sample_paths = ["0/1.png", "0/2.png", "1/3.png",...]
    >>>         super(ElasticMnistDataset, self).__init__(
    >>>         "train-ds",
    >>>         len(self.sample_paths),
    >>>         32,
    >>>         1,
    >>>         True,
    >>>     )
    >>>
    >>>     def read_sample(self, index):
    >>>         path = self.sample_paths[index]
    >>>         image = cv2.imread(image_path)
    >>>         return image
    >>>
    >>> dataset = MyElasticDataset()
    >>> state = dataset.state_dict()  # export checkpoint
    >>> dataset.load_state_dict(state)  # load checkpoint
    >>> data_loader = DataLoader(
    >>>     dataset=dataset, batch_size=args.batch_size, num_workers=2,
    >>> )

    Args:
        name: str, the name of dataset.
        dataset_size: the number of samples in the dataset.
        batch_size: int, the size of batch samples to compute gradients
            in a trainer process.
        epochs: int, the number of epoch.
        shuffle: bool, whether to shuffle samples in the dataset.
        num_minibatches_per_shard: int, the number of mini-batch per shard.
    """

    def __init__(
        self,
        name,
        dataset_size,
        batch_size,
        epochs,
        shuffle=False,
        num_minibatches_per_shard=2,
    ):
        from dlrover.python.elastic_agent.sharding.client import IndexShardingClient

        self._shard_client = IndexShardingClient(
            dataset_name=name,
            batch_size=batch_size,
            num_epochs=epochs,
            dataset_size=dataset_size,
            shuffle=shuffle,
            num_minibatches_per_shard=num_minibatches_per_shard,
            storage_type="text",
        )

    def __len__(self):
        return self._shard_client.get_total_sample_num()

    def __getitem__(self, _):
        index = self._shard_client.fetch_sample_index()
        return self.read_sample(index)

    def report_batch_done(self, batch_size=None):
        """After updating models using the samples, the dataset need to
        report the batch completion."""
        self._shard_client.report_batch_done(batch_size)

    @abstractmethod
    def read_sample(self, index):
        """Implement to read sample data by the index."""
        pass


class SimpleElasticDataset(ElasticDataset):
    """
    Get data index from Dynamic Data Sharding Service, read data by index
    and process data by `data_process_fn`.
    """

    def __init__(
        self,
        name,
        data_process_fn,
        dataset_size,
        batch_size,
        epochs,
        shuffle,
        num_minibatches_per_shard,
    ):
        """
        Args:
            data_process_fn: Function for data IO and processing.
            path: str, the path of dataset meta file. For example, if the image
                is stored in a folder. The meta file should be a
                text file where each line is the absolute path of a image.
            batch_size: int, the size of batch samples to compute gradients
                in a trainer process.
            epochs: int, the number of epoch.
            shuffle: bool, whether to shuffle samples in the dataset.
            num_minibatches_per_shard: int, the number of mini-batch per shard.
        """
        super(SimpleElasticDataset, self).__init__(
            name=name,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            num_minibatches_per_shard=num_minibatches_per_shard,
        )
        self.data_process_fn = data_process_fn

    def read_sample(self, index):
        return self.data_process_fn(index)
