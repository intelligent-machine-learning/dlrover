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

import math
import random
from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.log_utils import default_logger as logger

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


class DatasetSplitter(metaclass=ABCMeta):
    """DatasetSplitter splits a dataset to shards.
    Attrtibutes:
        dataset_size: the number of records in the dataset.
        shard_size: the number of records in a shard.
        num_epochs: the number of passes of the entire dataset.
    """

    def __init__(self, dataset_size, shard_size, num_epochs) -> None:
        self._dataset_size = dataset_size
        self._shard_size = shard_size
        self._num_epochs = num_epochs

    @abstractmethod
    def create_shards(self):
        """Split the dataset to shards"""
        pass

    @abstractmethod
    def get_shards(self) -> List[Shard]:
        """Get all shards of the dataset"""
        pass


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

    def __init__(
        self,
        dataset_name: str,
        dataset_size: int,
        shard_size: int,
        num_epochs: int,
        shuffle=False,
        batch_size=None,
        max_shard_count=_MAX_SHARD_COUNT,
    ):
        super(TableDatasetSplitter, self).__init__(
            dataset_size,
            shard_size,
            num_epochs,
        )
        self._dataset_name = dataset_name
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._max_shard_count = max_shard_count
        self._epoch = 0
        self._subepoch_num_per_epoch = 0
        self._shards: List[Shard] = []
        self._subepoch_idx = 0

    def get_shards(self):
        return self._shards

    def create_shards(self, model_version=-1):
        logger.info(
            "Creating a new set of shards for dataset {} with epoch {}".format(
                self._dataset_name, self._epoch
            )
        )
        shard_count = math.ceil(self._dataset_size / self._shard_size)
        if shard_count <= self._max_shard_count:
            if not self._shards:
                self._shards = self._create_shards_with_range(
                    0, self._dataset_size
                )
            self._epoch += 1
        else:
            self._subepoch_num_per_epoch = math.ceil(
                shard_count / self._max_shard_count
            )
            if self._subepoch_idx >= self._subepoch_num_per_epoch:
                self._subepoch_idx = 0

            if self._subepoch_idx == 0:
                self._epoch += 1

            self._subepoch_idx += 1

            logger.info(
                "Creating tasks for dataset:{} in a subepoch, "
                "subepoch_idx:{}, subepoch_num:{}, epoch:{}".format(
                    self._dataset_name,
                    self._subepoch_idx,
                    self._subepoch_num_per_epoch,
                    self._epoch,
                )
            )

            subepoch_records = self._max_shard_count * self._shard_size
            start_idx = (self._subepoch_idx - 1) * subepoch_records
            end_idx = start_idx + subepoch_records
            if end_idx > self._dataset_size:
                end_idx = self._dataset_size
            self._shards = self._create_shards_with_range(start_idx, end_idx)
        if self._shuffle:
            random.shuffle(self._shards)

    def _create_shards_with_range(self, start_idx, end_idx) -> List[Shard]:
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
        logger.info(
            "Create %s shards with range [%s, %s) ",
            len(shards),
            start_idx,
            end_idx,
        )
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

    def __init__(
        self,
        dataset_name,
        dataset_size,
        shard_size,
        num_epochs,
        shuffle=False,
        batch_size=None,
    ):
        super(TextDatasetSplitter, self).__init__(
            dataset_size, shard_size, num_epochs
        )
        self._dataset_name = dataset_name
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._epoch = 0
        self._shards: List[Shard] = []

    def get_shards(self) -> List[Shard]:
        return self._shards

    def create_shards(self):
        self._shards = self._create_shards_with_indices(
            0,
            self._dataset_size,
        )
        self._epoch += 1

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
