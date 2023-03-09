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
import time

from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.shard.base_dataset_manager import (
    DatasetManger,
    DatasetShardCheckpoint,
    DoingTask,
    Task,
)
from dlrover.python.master.shard.dataset_splitter import DatasetSplitter, Shard

_MAX_TASK_RETRIES = 3


class BatchDatasetManager(DatasetManger):
    """BatchDatasetManager create tasks with shards in a static dataset.
    Attributes:
        task_type: the type of computation task like "training",
            "evaluation" and "prediction".
        batch_size: the size of a batch.
        dataset_splitter: DatasetSplitter instace to split the dataset
            into shards.
    """

    def __init__(
        self,
        task_type,
        batch_size,
        dataset_splitter: DatasetSplitter,
    ):
        super(BatchDatasetManager, self).__init__(
            task_type, batch_size, dataset_splitter
        )
        self._max_task_completed_time = 0
        self._task_id = 0
        self._completed_step = 0

    def get_task(self, node_type, node_id) -> Task:
        """Return next Task"""

        if not self.todo and not self._dataset_splitter.epoch_finished():
            # Start a new epoch
            # num_epochs <= 0 indicates that the master will create data
            # shards infinitely. So, the worker can use the dataset like
            # `dataset.repeat()`.
            shards = self._dataset_splitter.create_shards()
            self._create_todo_tasks(shards)
        if not self.todo:
            # No more tasks
            return Task.create_invalid_task()
        task: Task = self.todo.pop(0)
        self.doing[task.task_id] = DoingTask(
            task, node_type, node_id, int(time.time())
        )
        logger.info(
            "Assign task %s of dataset %s to %s %s",
            task.task_id,
            self._dataset_splitter.dataset_name,
            node_type,
            node_id,
        )
        return task

    def get_epoch(self):
        return self._dataset_splitter.get_epoch()

    def completed(self):
        return (
            self._dataset_splitter.epoch_finished()
            and not self.todo
            and not self.doing
        )

    def _create_todo_tasks(self, shards):
        tasks = []
        for shard in shards:
            task = Task(self._task_id, self._task_type, shard)
            tasks.append(task)
            self._task_id += 1

        logger.info(
            "todo.extend: %d tasks created for dataset = %s.",
            len(tasks),
            self._dataset_splitter.dataset_name,
        )
        self.todo.extend(tasks)

    def report_task_status(self, task_id, success):
        doing_task = self.doing.pop(task_id)
        if not doing_task:
            logger.warning(
                "Unknown task_id: %d of dataset %s"
                % (task_id, self._dataset_splitter.dataset_name)
            )
            success = False
        elif not success:
            logger.warning(
                "Task %d of %s failed "
                % (task_id, self._dataset_splitter.dataset_name)
            )
            self.recover_task(doing_task.task)
        else:
            self._update_completed_step(doing_task.task)
            logger.info(
                "Task:%d completed, %d remaining tasks for Dataset %s",
                task_id,
                len(self.todo) + len(self.doing),
                self._dataset_splitter.dataset_name,
            )
            task_completed_time = time.time() - doing_task.start_time
            if task_completed_time > self._max_task_completed_time:
                self._max_task_completed_time = task_completed_time
        return success, doing_task

    def _update_completed_step(self, task: Task):
        record_count = task.shard.end - task.shard.start
        batch_count = math.ceil(record_count / self._batch_size)
        self._completed_step += batch_count
        self._latest_task_end_time = int(time.time())

    def get_completed_step(self):
        return self._completed_step

    def recover_task(self, task):
        if not self._check_exceed_max_task_retries(task):
            self.todo.append(task)

    def _check_exceed_max_task_retries(self, task: Task):
        task.retry_count += 1
        if task.retry_count > _MAX_TASK_RETRIES:
            logger.error(
                "A task %s of failed with %d retries "
                % (task.shard.name, _MAX_TASK_RETRIES)
            )
            return True
        return False

    def get_doing_tasks(self):
        return self.doing

    def checkpoint(self):
        todo_shards = []
        for task in self.todo:
            todo_shards.append([task.shard.start, task.shard.end])

        doing_shards = []
        for task_id in self.doing:
            task = self.doing[task_id].task
            doing_shards.append([task.shard.start, task.shard.end])

        return DatasetShardCheckpoint(
            dataset_name=self._dataset_splitter.dataset_name,
            todo=todo_shards,
            doing=doing_shards,
            epoch=self._dataset_splitter.epoch,
        )

    def restore_checkpoint(self, checkpoint: DatasetShardCheckpoint):
        """Restore the task manager from a checkpoint"""
        self._dataset_splitter.epoch = checkpoint.epoch
        self.todo = []
        for shard_indices in checkpoint.doing + checkpoint.todo:
            shard = Shard(
                name=self._dataset_splitter.dataset_name,
                start=shard_indices[0],
                end=shard_indices[1],
            )
            self.todo.append(
                Task(
                    self._task_id,
                    self._task_type,
                    shard,
                )
            )
            self._task_id += 1
