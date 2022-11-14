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

from dlrover.python.master.shard_manager.dataset_splitter import Shard


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
        return Task(-1, "", Shard("", -1, -1))


class TaskManger(metaclass=ABCMeta):
    @abstractmethod
    def get_task(self, worker_id):
        """Return a task with a shard for the worker with worker_id."""
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
