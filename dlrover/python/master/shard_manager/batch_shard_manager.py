from dlrover.python.master.shard_manager.base_shard_manager import (
    ShardManger,
    Task,
)
from dlrover.python.master.shard_manager.dataset_splitter import DatasetSplitter


class DatasetShardManager(ShardManger):
    def __init__(
        self,
        dataset_splitter: DatasetSplitter,
    ):
        self._dataset_splitter = dataset_splitter

