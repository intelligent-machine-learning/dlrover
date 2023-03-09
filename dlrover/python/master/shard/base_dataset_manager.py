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

import json
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from dlrover.proto import elastic_training_pb2
from dlrover.python.master.shard.dataset_splitter import DatasetSplitter, Shard


class Task(object):
    """A task contains a shard and will be assigned to a worker.
    Attributes:
        task_id: int, the index of the task.
        task_type: str, task type may be "training", "evaluation"
            and "prediction".
        shard: a record shard.
        retry_count: the count to retry a task.
    """

    def __init__(self, task_id, task_type, shard: Shard):
        self.task_id = task_id
        self.task_type = task_type
        self.shard = shard
        self.retry_count = 0

    @classmethod
    def create_invalid_task(self):
        return Task(-1, elastic_training_pb2.NONE, Shard("", -1, -1))


class DoingTask(object):
    """DoingTask records which node fetches a task and when.
    Attributes:
        task: a task with a data shard.
        node_id: the id of a node.
        start_time: the timestamp of a worker to fetch the task.
    """

    def __init__(
        self, task: Task, node_type: str, node_id: int, start_time: int
    ):
        self.task = task
        self.node_type = node_type
        self.node_id = node_id
        self.start_time = start_time


class DatasetShardCheckpoint(object):
    def __init__(
        self,
        dataset_name,
        todo,
        doing,
        epoch,
        splitter=None,
    ):
        """
        TODO: support checkpoint for indices.
        Args:
            todo: [[start_0, end_0], [start_1, end_1]],
            doing: [[start_2, end_2], [start_3, end_3]],
            current_epoch: int64, the index of epoch,
            epoch: the epoch index of dataset.
        """

        self.dataset_name = dataset_name
        self.todo = todo
        self.doing = doing
        self.epoch = epoch
        self.splitter = splitter

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, checkpoint_str):
        checkpoint_dict = json.loads(checkpoint_str)
        return DatasetShardCheckpoint(**checkpoint_dict)


class DatasetManger(metaclass=ABCMeta):
    """DatasetManger manages the task with a shard of the dataset
    Attributes:
        todo: A list to store tasks.
        doing: Dict[int, DoingTask] where key is the task id.
    """

    def __init__(
        self, task_type, batch_size, dataset_splitter: DatasetSplitter
    ):
        self.todo: List[Task] = []
        self.doing: Dict[int, DoingTask] = {}

        self._task_type = task_type
        self._batch_size = batch_size
        self._dataset_splitter = dataset_splitter
        self._latest_task_end_time = 0

    def get_latest_task_end_time(self):
        return self._latest_task_end_time

    @abstractmethod
    def get_epoch(self):
        """Get the training epoch"""
        pass

    @abstractmethod
    def completed(self):
        """Check whether the dataset manager completes."""
        pass

    @abstractmethod
    def get_task(self, node_type, node_id) -> Task:
        """Return a task with a shard for the node with node_id."""
        pass

    @abstractmethod
    def recover_task(self, task):
        """Recover a dispatched task if a worker fails."""
        pass

    @abstractmethod
    def report_task_status(self, task_id, success):
        """The worker reports the status of the shard."""
        pass

    @abstractmethod
    def get_completed_step(self):
        """Get the completed step."""
        pass

    @abstractmethod
    def checkpoint(self):
        """Checkpoint uncompleted shards"""
        pass

    @abstractmethod
    def restore_checkpoint(self, checkpoint):
        """Restore uncompleted data shards from a checkpoint."""
        pass
