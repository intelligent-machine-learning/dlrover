# Copyright 2020 The DLRover Authors. All rights reserved.
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

import os
import sys
import threading
import time
from collections import OrderedDict
from multiprocessing import SimpleQueue

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.monitor.training import (
    TrainingProcessReporter,
)

training_reporter = TrainingProcessReporter()

_DEFAULT_MINI_BATCH_NUM_PER_SHARD = 10


class ShardingClient(object):
    """ShardingClient queries data shards from the DLRover master.
    Args:
        dataset_name: the name of dataset.
        batch_size: the size of batch data.
        num_epochs: the number of epochs.
        dataset_size: the size of dataset.
        shuffle: whether to shuffle shards.
        task_type: Task type is the computation type like
            elastic_training_pb2.TRAINING, elastic_training_pb2.EVALUATION.
        num_minibatches_per_shard: the number of batch in each shard.
        storage_type: the storage type of dataset. It is "text" if the
            dataset is stored in a text file. It is "table" if the
            dataset is stored in a table like MaxCompute and Hive.
    Example:
        batch_size = 64
        client = ShardingClient(
            datset_name="test",
            batch_size=batch_size,
            num_epochs=1,
            dataset_size=10000,
        )
        while True:
            shard = client.fetch_shard()
            if not shard:
                break
            for i in range(shard.start, shard.end):
                print(i)
                if i % batch_size == 0:
                    client.report_batch_done()
    """

    def __init__(
        self,
        dataset_name,
        batch_size,
        num_epochs,
        dataset_size,
        shuffle=False,
        task_type=elastic_training_pb2.TRAINING,
        num_minibatches_per_shard=_DEFAULT_MINI_BATCH_NUM_PER_SHARD,
        storage_type="",
    ):
        self._mc = GlobalMasterClient.MASTER_CLIENT
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dataset_size = dataset_size
        self._shuffle = shuffle
        self._task_type = task_type
        self._storage_type = storage_type
        self._num_minibatches_per_shard = num_minibatches_per_shard
        self._lock = threading.Lock()
        self._reported_record_count = {}
        self._current_task = None
        self._pending_tasks = OrderedDict()
        self._dataset_name = dataset_name
        self._batch_count = 0
        self._max_shard_count = sys.maxsize
        self._shard_count = 0
        self._report_sharding_params()

    def _report_sharding_params(self):
        if self._num_epochs and self._dataset_size:
            self._mc.report_dataset_shard_params(
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                dataset_size=self._dataset_size,
                shuffle=self._shuffle,
                num_minibatches_per_shard=self._num_minibatches_per_shard,
                dataset_name=self._dataset_name,
                task_type=self._task_type,
                storage_type=self._storage_type,
            )

    def get_minibatch_count_per_epoch(self):
        return self._dataset_size // self._batch_size

    def reset_dataset(self):
        # Only dataset with a name will be reset.
        self._mc.reset_dataset(self._dataset_name)

    def get_current_task(self):
        return self._current_task

    def get_task(self) -> elastic_training_pb2.Task:
        training_reporter.set_start_time()
        if self._shard_count >= self._max_shard_count:
            return None
        for _ in range(5):
            success, task = self._mc.get_task(self._dataset_name)
            if success:
                break
            time.sleep(5)
        if task.shard.end - task.shard.start > 0:
            with self._lock:
                self._pending_tasks[task.task_id] = task
                if len(self._pending_tasks) == 1:
                    self._current_task = task
            self._shard_count += 1
            return task
        return None

    def _report_task(self, task, err_msg=""):
        self._mc.report_task_result(
            self._dataset_name,
            task.task_id,
            err_msg,
        )

    def report_all_task_error(self, err_msg):
        while self._pending_tasks:
            _, task = self._pending_tasks.popitem()
            self._report_task(task, err_msg)

    def report_batch_done(self, batch_size=None, err_msg="", task_ids=[]):
        """
        Report the number of records in the latest processed batch,
        so DynamicShardingManager knows if some pending tasks are finished
        and report_task_result to the master.
        Return True if there are some finished tasks, False otherwise.
        """
        reported = False
        if not self._pending_tasks:
            return reported
        record_count = batch_size if batch_size else self._batch_size
        self._batch_count += 1

        with self._lock:
            if not task_ids:
                task_ids = list(self._pending_tasks.keys())
            for task_id in task_ids:
                if record_count > 0:
                    task = self._pending_tasks[task_id]
                    task_record_count = task.shard.end - task.shard.start

                    self._reported_record_count.setdefault(task_id, 0)
                    cur_count = self._reported_record_count[task_id]
                    if cur_count + record_count >= task_record_count:
                        self._report_task(task, err_msg)
                        reported = True
                        self._reported_record_count.pop(task_id)
                        self._pending_tasks.pop(task_id)
                        record_count = (
                            cur_count + record_count - task_record_count
                        )
                        self._report_training_local_step()
                    else:
                        self._reported_record_count[task_id] += record_count
                        record_count = 0

        if self._pending_tasks:
            self._current_task = next(iter(self._pending_tasks.values()))
        return reported

    def _report_training_local_step(self):
        if not training_reporter.called_in_tf_hook:
            training_reporter.report_resource_with_step(self._batch_count)

    def fetch_shard(self):
        """Fetch data shard and each shard contains the name,
        start and end index.
        """
        task = self.get_task()
        if task:
            return task.shard
        return None

    def get_shard_checkpoint(self):
        """Get the data shard checkpoint of a dataset.
        If the dataset is None, returns the checkpoint of training dataset.

        Args:
            dataset_name: string.

        Returns:
            Json String: {
                "dataset_name": string.
                "todo": [(start_0, end_0), (start_1, end_1)],
                "doing": [(start_2, end_2), (start_3, end_3)],
                "current_epoch": int64, the index of epoch,
                "num_epochs": int64, the number of epoch,
                "batch_size": int64, batch size,
                "dataset_size": int64, the size of dataset,
                "shuffle_shards": bool, true of false.
                ""
            }
        """
        shard_checkpoint = self._mc.get_shard_checkpoint(self._dataset_name)
        return shard_checkpoint.content

    def restore_shard_from_checkpoint(self, shard_checkpoint):
        res = self._mc.report_shard_checkpoint(shard_checkpoint)
        return res.success

    def get_current_epoch(self):
        res = self._mc.get_dataset_epoch(self._dataset_name)
        return res.epoch - 1

    def get_total_sample_num(self):
        return self._dataset_size * self._num_epochs

    def set_max_shard_count(self):
        world_size = int(os.getenv("WORLD_SIZE", 0))
        if world_size:
            total_shard_count = self._mc.get_dataset_shard_num(
                self._dataset_name
            )
            if total_shard_count == 0:
                return
            self._max_shard_count = total_shard_count // world_size
            logger.info(
                "The max number of shards is %s", self._max_shard_count
            )


class IndexShardingClient(ShardingClient):
    """ShardingClient queries data shards from the DLRover master
    and generates the index of sample from the shard.
    Users can read data from the disk by the sample index.
    Args:
        dataset_name: the name of dataset.
        batch_size: the size of batch data.
        num_epochs: the number of epochs.
        dataset_size: the size of dataset.
        shuffle: whether to shuffle shards.
        task_type: Task type is the computation type like
            elastic_training_pb2.TRAINING, elastic_training_pb2.EVALUATION.
        num_minibatches_per_shard: the number of batch in each shard.
        storage_type: the storage type of dataset. It is "text" if the
            dataset is stored in a text file. It is "table" if the
            dataset is stored in a table like MaxCompute and Hive.
        num_workers: the number of worker processes to share the client
            to get the sample index.
    """

    def __init__(
        self,
        dataset_name,
        batch_size,
        num_epochs,
        dataset_size,
        shuffle=False,
        task_type=elastic_training_pb2.TRAINING,
        num_minibatches_per_shard=_DEFAULT_MINI_BATCH_NUM_PER_SHARD,
        storage_type="",
        num_workers=1,
    ):
        super(IndexShardingClient, self).__init__(
            dataset_name,
            batch_size,
            num_epochs,
            dataset_size,
            shuffle,
            task_type,
            num_minibatches_per_shard,
            storage_type,
        )
        self._num_workers = num_workers
        self._sample_queue = SimpleQueue()
        self._report_sharding_params()

        threading.Thread(
            target=self._prefetch_sample_indices,
            name="fetch_sample_indices",
            daemon=True,
        ).start()

    def _prefetch_sample_indices(self):
        while True:
            if self._sample_queue.empty():
                task = self.get_task()
                if not task or not task.shard:
                    for _ in range(128):
                        self._sample_queue.put(None)
                    break
                ids = (
                    task.shard.indices
                    if task.shard.indices
                    else list(range(task.shard.start, task.shard.end))
                )
                for i in ids:
                    self._sample_queue.put(i)
            else:
                time.sleep(0.001)

    def fetch_sample_index(self):
        """Fetch an index of the sample. The function get an index
        from a queue because there may be multiple sub-process to call
        the function.
        """
        index = self._sample_queue.get()
        if index is None:
            logger.info("No more data.")
            raise StopIteration()
        return index

    def clear_shard_queue(self):
        self._sample_queue = SimpleQueue()

    def restore_shard_from_checkpoint(self, shard_checkpoint):
        # To avoid duplicate shards, drop all shards in the _shard_queue
        # before restoring shard from checkpoint
        # self.clear_shard_queue()
        super().restore_shard_from_checkpoint(shard_checkpoint)
