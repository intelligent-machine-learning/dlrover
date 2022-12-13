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

from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.constants import DatasetType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.watcher.base_watcher import Node


class CustomMetricKey(object):
    INIT_TRAINING_TIME = "init_training_time"


class TrainingHyperParams(object):
    def __init__(self, batch_size=0, epoch=0, max_steps=0):
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_steps = max_steps


class DatasetMetric(metaclass=ABCMeta):
    """DatasetMetric stores the meta of the dataset.
    Attributed:
        name: the name of dataset.
        size: the number of records in the dataset.
        storage_size: the storage size of the dataset.
    """

    def __init__(self, name, size=0, storage_size=0):
        self.name = name
        self.size = size
        self.storage_size = storage_size

    @abstractmethod
    def get_name(self):
        """Get the name of dataset.
        Returns:
            name: String
        """
        pass

    @abstractmethod
    def get_size(self):
        """Get the total number of samples in the dataset.
        Returns:
            size: int
        """
        pass

    @abstractmethod
    def get_storage_size(self):
        """Get the total physical storage size of the dataset.
        Returns:
            storage_size: int
        """
        pass

    @classmethod
    def new_dataset_metric(cls, ds_type, name, size):
        if not ds_type or ds_type == DatasetType.TEXT:
            return TextDatasetMetric(name, size)
        else:
            logger.warning("Not support dataset type %s", ds_type)


class TextDatasetMetric(DatasetMetric):
    """TextDatasetMetric contains metrics of a text dataset.
    Attributes:
        name: the path of the text file.
    """

    def __init__(
        self,
        name,
        size=0,
    ):
        super(TextDatasetMetric, self).__init__(name, size)

    def get_name(self):
        return self.name

    def get_size(self):
        if self.size > 0:
            return self.size
        try:
            if self.name:
                count = 0
                for count, _ in enumerate(open(self.name, "r")):
                    pass
                self.size = count
        except Exception as e:
            logger.error(e)
        return self.size

    def get_storage_size(self):
        return self.storage_size


class TensorStats(object):
    """TensorStats contains tensor statistics of a deep learning model"""

    def __init__(self, variable_count, total_variable_size, max_variable_size):
        self.variable_count = variable_count
        self.total_variable_size = total_variable_size
        self.max_variable_size = max_variable_size


class OpStats(object):
    """TensorStats contains OP statistics of a deep learning model"""

    def __init__(self, op_count, update_op_count, input_fetch_dur, flops):
        self.op_count = op_count
        self.update_op_count = update_op_count
        self.input_fetch_dur = input_fetch_dur
        self.flops = flops


class ModelMetric(object):
    """ModelMetric contains profiling data of a model."""

    def __init__(self, tensor_stats: TensorStats, op_stats: OpStats):
        self.tensor_stats = tensor_stats
        self.op_stats = op_stats


class RuntimeMetric(object):
    """RuntimeMetric contains the runtime statistics of a job."""

    def __init__(
        self, running_nodes: List[Node], global_step=0, speed=0, timestamp=0
    ):
        self.running_nodes = running_nodes
        self.global_step = global_step
        self.speed = speed
        self.timestamp = timestamp

    def clear(self):
        self.running_nodes = []
