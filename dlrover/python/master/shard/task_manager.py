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

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Dict, List

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.constants import NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.shard.base_dataset_manager import (
    DatasetManger,
    DatasetShardCheckpoint,
)
from dlrover.python.master.shard.batch_dataset_manager import (
    BatchDatasetManager,
)
from dlrover.python.master.shard.dataset_splitter import DatasetSplitter

_TASK_TIMEOUT_THRESHOLD_SECS = 1800


class TaskManager(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(
        self, worker_restart_timeout: int, speed_monitor: SpeedMonitor
    ):
        """
        Args:
            worker_restart_timeout: Whether to relaunch a worker
                when it does not report a task status for a long time.
            speed_monitor: monitor the training speed with workers.
        """
        self._lock = threading.Lock()
        self._worker_restart_timeout = worker_restart_timeout
        self._should_stop = False
        self._datasets: Dict[str, DatasetManger] = OrderedDict()
        self._worker_start_task_time: Dict[int, float] = {}
        self._task_timeout_callbacks: List[Callable] = []
        self._speed_monitor = speed_monitor

    def new_dataset(
        self,
        batch_size,
        dataset_size,
        dataset_name,
        dataset_splitter: DatasetSplitter,
        task_type=elastic_training_pb2.NONE,
    ):
        logger.info(
            f"New {task_type} dataset {dataset_name} with, "
            f"batch size = {batch_size} dataset size = {dataset_size}"
        )

        with self._lock:
            if dataset_name in self._datasets:
                logger.info(
                    "The shards for dataset %s have already been initialized. "
                    "Ignore these shard parameters.",
                    dataset_name,
                )
                return
            if dataset_size < 0:
                logger.error(
                    "No shard for datset %s because dataset size %s <= 0",
                    dataset_name,
                    dataset_size,
                )
                return
            dataset = BatchDatasetManager(
                task_type=task_type,
                batch_size=batch_size,
                dataset_splitter=dataset_splitter,
            )
            self._datasets[dataset_name] = dataset

    def get_dataset_task(self, node_type, node_id, dataset_name):
        """Return next Task"""
        with self._lock:
            dataset = self._datasets.get(dataset_name, None)
            if dataset:
                task = dataset.get_task(node_type, node_id)
                if (
                    task.task_type == elastic_training_pb2.EVALUATION
                    and node_type == NodeType.WORKER
                ):
                    # All workers will stop training to evaluate the model
                    # at parallel validation
                    logger.info(
                        "Reset speed monitor if the worker starts evaluation"
                    )
                    self._speed_monitor.reset_running_speed_monitor()
                    self._speed_monitor.set_worker_start_eval_time(node_id)
                if task.task_type == elastic_training_pb2.TRAINING:
                    self._speed_monitor.add_running_worker(node_type, node_id)
                    self._speed_monitor.update_worker_eval_time(node_id)
                self._worker_start_task_time[node_id] = time.time()
                return task
            else:
                return None

    def get_dataset(self, dataset_name):
        return self._datasets.get(dataset_name, None)

    def report_dataset_task(self, request, success):
        """Report if the task is successful or not"""

        task_id = request.task_id
        dataset_name = request.dataset_name
        with self._lock:
            dataset = self._datasets.get(dataset_name, None)
            if not dataset:
                raise ValueError(
                    "There is no dataset shard for the dataset {}".format(
                        dataset_name
                    )
                )
            success, doing_task = dataset.report_task_status(task_id, success)
            if success:
                self._worker_start_task_time[doing_task.node_id] = time.time()
                return doing_task.task, doing_task.node_id
            return None, None

    def task_hanged(self):
        dataset_hang = []
        for _, ds in self._datasets.items():
            end_time = ds.get_latest_task_end_time()
            hang = (
                end_time > 0
                and time.time() - end_time > _TASK_TIMEOUT_THRESHOLD_SECS
            )
            dataset_hang.append(hang)
        if dataset_hang:
            return all(dataset_hang)
        return False

    def finished(self):
        """Return if all tasks are done"""
        if not self._datasets:
            return False
        finished = all([ds.completed() for ds in self._datasets.values()])
        return finished

    def recover_tasks(self, node_type, node_id):
        """Recover doing tasks for a dead worker if needed"""
        for name, dataset in self._datasets.items():
            doing_tasks = dataset.doing
            if not doing_tasks:
                continue
            ids = [
                task_id
                for task_id, doing_task in doing_tasks.items()
                if doing_task.node_id == node_id
                and doing_task.node_type == node_type
            ]
            if not ids:
                continue
            request = elastic_training_pb2.ReportTaskResultRequest()
            recover_tasks = []
            for id in ids:
                request.task_id = id
                request.dataset_name = name
                recover_tasks.append(id)
                self.report_dataset_task(request, False)
            logger.info(
                "Recover tasks %s of dataset %s assigned to %s-%d",
                recover_tasks,
                name,
                node_type,
                node_id,
            )

    def start(self):
        if self._worker_restart_timeout > 0:
            threading.Thread(
                target=self._check_and_reassign_timeout_tasks,
                name="check_timeout_tasks",
                daemon=True,
            ).start()

    def reset_worker_start_task_time(self, worker_id):
        self._worker_start_task_time[worker_id] = time.time()

    def set_task_timeout_callback(self, callback_fn):
        self._task_timeout_callbacks.append(callback_fn)

    def _invoke_task_timeout_callback(self, worker_id):
        for callback_fn in self._task_timeout_callbacks:
            callback_fn(worker_id)

    def _check_and_reassign_timeout_tasks(self):
        """Check whether there are timeout tasks periodically."""
        logger.info("Start the thread to monitor timeout tasks.")
        while True:
            for _, dataset in self._datasets.items():
                # Copy doing task list because the doing list will pop items
                # in the following loop.
                doing_tasks = dataset.doing.copy()
                cur = time.time()
                for task_id, doing_task in doing_tasks.items():
                    start = self._worker_start_task_time.get(
                        doing_task.node_id, cur
                    )
                    if (
                        doing_task.task.task_type
                        == elastic_training_pb2.EVALUATION
                        and cur - start
                        > max(
                            _TASK_TIMEOUT_THRESHOLD_SECS,
                            self._worker_restart_timeout,
                        )
                    ):
                        logger.info(
                            "worker %d timeout with task %d, relaunch it",
                            doing_task.node_id,
                            task_id,
                        )
                        dataset.report_task_status(task_id, success=False)
                        self._invoke_task_timeout_callback(doing_task.node_id)
                        break
            time.sleep(30)

    def get_dataset_checkpoint(self, dataset_name):
        """Get the data shard checkpoint by dataset name.

        Args:
            dataset_name: string

        Returns:
            DatasetShardCheckpoint.
        """
        with self._lock:
            if dataset_name in self._datasets:
                dataset = self._datasets[dataset_name]
                return dataset.checkpoint()
            else:
                return None

    def restore_dataset_from_checkpoint(self, checkpoint):
        try:
            dataset_checkpoint = DatasetShardCheckpoint.from_json(checkpoint)
            dataset = self._datasets.get(dataset_checkpoint.dataset_name, None)
            if not dataset:
                logger.error("No dataset for checkpoint %s", checkpoint)

            dataset.restore_checkpoint(dataset_checkpoint)
            logger.info(
                "Restore %s dataset with %s shards from checkpoint",
                dataset_checkpoint.dataset_name,
                len(dataset.todo) + len(dataset.doing),
            )
            return True
        except Exception as e:
            logger.error("Fail to restore shards from the checkpoint %s", e)

        return False

    def get_dataset_epoch(self, dataset_name):
        if dataset_name in self._datasets:
            return self._datasets[dataset_name].get_epoch()
        else:
            logger.error("There is not exit dataset {}".format(dataset_name))
            return 0

    def training_started(self):
        """The training has started if there is a completed batch"""
        for _, dataset in self._datasets.items():
            if dataset.get_completed_step() > 0:
                return True
        return False
