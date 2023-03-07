from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from dlrover.python.elastic_agent.sharding.client import IndexShardingClient


def count_line(path):
    with open(path, "r") as fp:
        for count, _ in enumerate(fp):
            pass
    return count


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
        dataset_size = count_line(path)

        self._shard_client = IndexShardingClient(
            dataset_name=path,
            batch_size=batch_size,
            num_epochs=epochs,
            dataset_size=dataset_size,
            shuffle=shuffle,
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
