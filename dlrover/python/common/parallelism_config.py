from dataclasses import dataclass
from typing import Optional
from dlrover.python.common.serialize import JsonSerializable


@dataclass
class DataLoaderConfig(JsonSerializable):
    """The configured parameters of DataLoader.
    Attr:
        name: a DataLoader instance has an unique name in a job.
        batch_size: the number of samples in a batch.
        num_workers: how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
        pin_memory: If True, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them.
    """
    name: str
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = False


@dataclass
class OptimizerConfig(JsonSerializable):
    name: str
    learning_rate: Optional[float] = None


@dataclass
class ParallelismConfig(JsonSerializable):
    dataloader: Optional[DataLoaderConfig] = None
    optimizer: Optional[DataLoaderConfig] = None
