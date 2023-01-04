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

import copy
import inspect
import math
import random
from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.log import default_logger as logger

_MAX_SHARD_COUNT = 50000


class Shard(object):
    """Shard contains index range or indices of some batch records.
    Attribute:
        name: str, the storage name of dataset which may be a local path of
            a file or a table name in a remote database.
        start: int, the start record index of the shard.
        end: int, the end record index of the shard.
        record_indices: indices of records in the dataset.
    """

    def __init__(self, name, start, end, record_indices: List[int] = None):
        self.name = name
        self.start = start
        self.end = end
        self.record_indices = record_indices


class PartitionOffsets(object):
    """PartitionOffsets contains index of the partition and its
       corresponding offsets.
    Attribute:
        partition_offsets: dict, key is the partition_index and
            value is the start offset of unconsumed sample
        partitions: list, the set of partition index
        partition_num: int, the number of partitions
    """

    def __init__(self, partition_offsets):
        self.partition_offsets = partition_offsets
        self.start_partition = 0
        self.partitions = []
        self.partition_num = 0
        self.partition_name_index = {}
        self._update_partitions()

    def _update_partitions(self):
        self.partitions = list(self.partition_offsets.keys())
        self.partition_num = len(self.partitions)
        self.partition_name_index = {
            k: v for k, v in enumerate(self.partitions)
        }

    def get_partition_index_by_name(self, partition_name):
        return self.partition_name_index.get(partition_name, None)

    def get_partitions(self):
        return self.partitions

    def get_round_robin_partition(self):
        index = self.start_partition % self.partition_num
        partition = self.partitions[index]
        self.start_partition += 1
        return partition

    def get_partition_offset(self, partition_index):
        return self.partition_offsets.get(partition_index, None)

    def set_partition_offset(self, partition_index, offset):
        self.partition_offsets[partition_index] = offset

    def to_dict(self):
        return self.partition_offsets


class DatasetSplitter(metaclass=ABCMeta):
    """DatasetSplitter splits a dataset to shards.
    Attrtibutes:
        dataset_name: the name of dataset.
        dataset_size: the number of records in the dataset.
        shard_size: the number of records in a shard.
        num_epochs: the number of passes of the entire dataset.
        epoch: the epoch index of the dataset.
    """

    def __init__(
        self, dataset_name, dataset_size, shard_size, num_epochs
    ) -> None:
        self.dataset_name = dataset_name
        self.epoch = 0
        self._dataset_size = dataset_size
        self._shard_size = shard_size
        self._num_epochs = num_epochs

    @abstractmethod
    def get_epoch(self):
        """Get the current epoch"""
        pass

    @abstractmethod
    def create_shards(self):
        """Split the dataset to shards"""
        pass

    @abstractmethod
    def get_shards(self) -> List[Shard]:
        """Get all shards of the dataset"""
        pass

    def epoch_finished(self) -> bool:
        """Check wether to finish the configured epochs"""
        return self.epoch >= self._num_epochs


class TableDatasetSplitter(DatasetSplitter):
    """TableDatasetSplitter split a dataset stored in a table like Hive or
    MaxCompute (ODPS) table. We can read data by record indices in the table.
    The shard contains index ranges [start, end) of batch records.
    Attributes:
        dataset_name: the name of the table.
        shuffle: whether to shuffle shards of the dataset.
        batch_size: the number of records in a batch.
        max_shard_count: the max number of shards in the memory.
            The value can limit the number of shards in the memory
            to avoid OOM.
    """

    STORAGE_TYPE = "table"

    def __init__(
        self,
        dataset_name: str,
        dataset_size: int,
        shard_size: int,
        num_epochs: int,
        shuffle=False,
        max_shard_count=_MAX_SHARD_COUNT,
    ):
        super(TableDatasetSplitter, self).__init__(
            dataset_name,
            dataset_size,
            shard_size,
            num_epochs,
        )
        self._dataset_name = dataset_name
        self._shuffle = shuffle
        self._max_shard_count = max_shard_count
        self._subepoch_num_per_epoch = 0
        self._shards: List[Shard] = []
        self._split_epoch_for_huge_dataset()

    def _split_epoch_for_huge_dataset(self):
        shard_count = math.ceil(self._dataset_size / self._shard_size)
        if shard_count > self._max_shard_count:
            self._subepoch_num_per_epoch = math.ceil(
                shard_count / self._max_shard_count
            )
            self._num_epochs = self._num_epochs * self._subepoch_num_per_epoch

    def get_epoch(self):
        if self._subepoch_num_per_epoch > 0:
            return int(self.epoch / self._subepoch_num_per_epoch)
        else:
            return self.epoch

    def get_shards(self):
        return self._shards

    def create_shards(self):
        logger.info(
            "Creating a new set of shards for dataset {} with epoch {}".format(
                self._dataset_name, self.epoch
            )
        )
        shard_count = math.ceil(self._dataset_size / self._shard_size)
        if shard_count <= self._max_shard_count:
            if not self._shards:
                self._shards = self._create_shards_with_range(
                    0, self._dataset_size
                )
        else:
            subepoch_idx = self.epoch % self._subepoch_num_per_epoch
            logger.info(
                "Creating tasks for dataset:%s in a subepoch, "
                "subepoch_idx:%s, subepoch_num:%s, epoch:%s",
                self._dataset_name,
                subepoch_idx,
                self._subepoch_num_per_epoch,
                int(self.epoch / self._subepoch_num_per_epoch),
            )

            subepoch_size = self._max_shard_count * self._shard_size
            start_idx = subepoch_idx * subepoch_size
            end_idx = start_idx + subepoch_size
            if end_idx > self._dataset_size:
                end_idx = self._dataset_size
            self._shards = self._create_shards_with_range(start_idx, end_idx)
        if self._shuffle:
            random.shuffle(self._shards)
        self.epoch += 1
        return self._shards

    def _create_shards_with_range(self, start_idx, end_idx) -> List[Shard]:
        logger.info("Create shard with range [%s, %s)", start_idx, end_idx)
        shards = []
        num_shards = (end_idx - start_idx) // self._shard_size
        for _ in range(num_shards):
            shard = Shard(
                name=self._dataset_name,
                start=start_idx,
                end=start_idx + self._shard_size,
            )
            shards.append(shard)
            start_idx += self._shard_size
        # Create a shard with the last records
        num_records_left = (end_idx - start_idx) % self._shard_size
        if num_records_left != 0:
            shard = Shard(
                name=self._dataset_name,
                start=start_idx,
                end=start_idx + num_records_left,
            )
            shards.append(shard)
        logger.info("%s shards are created ", len(shards))
        return shards


class TextDatasetSplitter(DatasetSplitter):
    """In a text dataset, each line is the location of sample.
    TextDatasetSplitter splits line numbers into shards and
    each shard contains many line numbers like {1, 5, 8, 2, 6, ...}
    Attributes:
        dataset_name: the path of the text file.
        shuffle: whether to shuffle samples of the dataset.
        batch_size: the number of records in a batch.
    """

    STORAGE_TYPE = "text"

    def __init__(
        self,
        dataset_name,
        dataset_size,
        shard_size,
        num_epochs,
        shuffle=False,
    ):
        super(TextDatasetSplitter, self).__init__(
            dataset_name, dataset_size, shard_size, num_epochs
        )
        self._dataset_name = dataset_name
        self._shuffle = shuffle
        self._shards: List[Shard] = []

    def get_epoch(self):
        return self.epoch

    def get_shards(self) -> List[Shard]:
        return self._shards

    def create_shards(self):
        self._shards = self._create_shards_with_indices(
            0,
            self._dataset_size,
        )
        self.epoch += 1
        return self._shards

    def _create_shards_with_indices(self, start_idx, end_idx) -> List[Shard]:
        shards = []
        record_indices = list(range(self._dataset_size))
        if self._shuffle:
            random.shuffle(record_indices)
        for shard_start_idx in range(start_idx, end_idx, self._shard_size):
            shard_end_idx = min(
                shard_start_idx + self._shard_size,
                end_idx,
            )
            size = shard_end_idx - shard_start_idx
            shard_indices = record_indices[0:size]
            record_indices = record_indices[size:]
            shards.append(
                Shard(
                    name=self._dataset_name,
                    start=shard_start_idx,
                    end=shard_end_idx,
                    record_indices=shard_indices,
                )
            )
        return shards


def new_dataset_splitter(
    shuffle,
    shard_size,
    dataset_size,
    num_epochs,
    dataset_name,
    storage_type=None,
):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.info(
        "New a datast splitter with: %s",
        [(i, values[i]) for i in args],
    )
    if not storage_type or storage_type == TableDatasetSplitter.STORAGE_TYPE:
        return TableDatasetSplitter(
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            shard_size=shard_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
        )
    elif storage_type == TextDatasetSplitter.STORAGE_TYPE:
        return TextDatasetSplitter(
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            shard_size=shard_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
        )
    else:
        raise ValueError("Not support dataset storage %s", storage_type)


class StreamingDatasetSplitter(DatasetSplitter):
    """StreamingDatasetSplitter split a dataset stored in a message queue like
    Kafka or SLS. We can read data by record offset in the logstore.
    The shard contains index ranges [start, end) and partition of records.
    Attributes:
        dataset_name: the name of the logstore.
        shuffle: whether to shuffle shards of the dataset, by default is False
        batch_size: the number of records in a batch.
        data_size: the number of records of the dataset. When
            data_size is set to -1,the number of records of the
            dataset is infinity.
        max_shard_count: the max number of shards in the memory.
            The value can limit the number of shards in the memory
            to avoid OOM.
    """

    STORAGE_TYPE = "sls"

    def __init__(
        self,
        dataset_name,
        shard_size,
        partition_offset,
        num_epochs=1,
        dataset_size=-1,
        shuffle=False,
        max_shard_count=None,
        fetch_data_size=10000,
    ):
        super(StreamingDatasetSplitter, self).__init__(
            dataset_name,
            dataset_size,
            shard_size,
            num_epochs,
        )
        self._dataset_name = dataset_name
        self._shard_size = shard_size
        self._partition_offset = partition_offset
        self.num_epochs = num_epochs
        self._dataset_size = dataset_size
        self._shuffle = shuffle
        self._max_shard_count = max_shard_count
        self._fetch_data_size = fetch_data_size
        self._shards = []
        self.epoch = 0

    def epoch_finished(self):
        finished = False
        if self._dataset_size == 0:
            finished = True
        return finished

    def get_epoch(self):
        return 1

    def to_checkpoint(self):
        partition_offset = self._partition_offset.to_dict()
        checkpoint = self.__dict__
        checkpoint.update({"_partition_offset": partition_offset})
        return checkpoint

    @staticmethod
    def from_checkpoint(checkpoint):
        init_args = {}
        args_key = [
            "dataset_name",
            "shard_size",
            "partition_offset",
            "num_epochs",
            "dataset_size",
            "shuffle",
            "max_shard_count",
            "fetch_data_size",
        ]
        partition_offset = PartitionOffsets(
            checkpoint.pop("_partition_offset")
        )
        for k in args_key:
            checkpoint_key = "_" + k
            init_args[k] = checkpoint.get(checkpoint_key)
        init_args["partition_offset"] = partition_offset
        return StreamingDatasetSplitter(**init_args)

    def get_shards(self):
        return self._shards

    def create_shards(self):
        logger.info(
            "Creating a new set of shards for dataset {} with epoch {}".format(
                self._dataset_name, self.epoch
            )
        )
        self._create_shards_with_range()
        return self._shards

    def get_partition_offset(self):
        return self._partition_offset

    def _create_shards_with_range(self):
        shards = []
        prev_partition_offset = copy.deepcopy(self._partition_offset)
        if self._dataset_size == -1:
            shard_count = self._fetch_data_size / self._shard_size
        else:
            shard_count = self._dataset_size / self._shard_size
        for i in range(int(shard_count)):
            partition_name = self._partition_offset.get_round_robin_partition()
            start = self._partition_offset.get_partition_offset(partition_name)
            end = start + self._shard_size
            shard = Shard(name=partition_name, start=start, end=end)
            self._shards.append(shard)
            start = end
            if self._dataset_size != -1:
                self._dataset_size = self._dataset_size - self._shard_size
            self._partition_offset.set_partition_offset(partition_name, end)
        logger.info("Create %s shards", len(shards))
        for p in self._partition_offset.get_partitions():
            logger.info(
                "partition %s : with range [%s, %s)",
                p,
                prev_partition_offset.get_partition_offset(p),
                self._partition_offset.get_partition_offset(p),
            )
