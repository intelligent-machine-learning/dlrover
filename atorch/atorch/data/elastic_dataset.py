from abc import abstractmethod
from typing import Any

from atorch.common.log_utils import default_logger as logger

try:
    from dlrover.trainer.torch import elastic_dataset as dlrover_ds

    BaseElasticDataset: Any = dlrover_ds.ElasticDataset
except ImportError:
    logger.warning("Please install dlrover[torch] to use elastic dataset.")
    BaseElasticDataset = object


class ElasticDataset(BaseElasticDataset):
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
        super(ElasticDataset, self).__init__(
            name=name,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            num_minibatches_per_shard=num_minibatches_per_shard,
        )

    @abstractmethod
    def read_sample(self, index):
        """Implement to read and process sample data by the index.
        For example, we can build a list with the link or path of
        all samples. Then, we can get the link from the list by
        an index and read the sample with the link. Users may need
        to build the list with samples when using ElasticDataset.
        """
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
