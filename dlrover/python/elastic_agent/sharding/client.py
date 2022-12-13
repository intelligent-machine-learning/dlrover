# Copyright 2020 The ElasticDL Authors. All rights reserved.
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

import threading
import time
from collections import OrderedDict

from dlrover.proto import elastic_training_pb2
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.monitor.training import (
    TrainingProcessReporter,
)

training_reporter = TrainingProcessReporter()


class ShardingClient(object):
    def __init__(
        self,
        dataset_name,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        task_type=elastic_training_pb2.TRAINING,
        num_minibatches_per_shard=0,
        master_client=None,
        storage_type="",
    ):
        self._mc = (
            master_client
            if master_client
            else GlobalMasterClient.MASTER_CLIENT
        )
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

    def get_task(self):
        training_reporter.set_start_time()
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
        return res.epoch
