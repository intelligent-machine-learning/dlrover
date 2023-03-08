# Copyright 2021 The DLRover Authors. All rights reserved.
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
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from dlrover.python.common.constants import MemoryUnit
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.stats.reporter import JobMeta, StatsReporter
from dlrover.python.master.stats.training_metrics import (
    CustomMetricKey,
    DatasetMetric,
    DatasetType,
    ModelMetric,
    OpStats,
    RuntimeMetric,
    TensorStats,
    TrainingHyperParams,
)
from dlrover.python.master.watcher.base_watcher import Node


class BaseMetricCollector(metaclass=ABCMeta):
    def __init__(self, job_meta: JobMeta, reporter=None):
        self._job_meta = job_meta
        self._stats_reporter = StatsReporter.new_stats_reporter(
            job_meta, reporter
        )

    @classmethod
    def catch_exception(cls, func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Fail to %s.%s by %s",
                    self.__class__.__name__,
                    func.__name__,
                    e,
                )

        return wrapper

    @abstractmethod
    def collect_model_metric(self, *args, **kwargs):
        pass

    @abstractmethod
    def collect_training_hyper_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def collect_dataset_metric(self, *args, **kwargs):
        pass

    @abstractmethod
    def collect_runtime_stats(
        self, speed_monitor: SpeedMonitor, running_nodes: List[Node]
    ):
        pass


class JobMetricCollector(BaseMetricCollector):
    """The collector receives model parameters message from workers
    and processes the message. Then, it will report model parameters
    to EasyDL server.
    """

    def __init__(self, job_uuid, namespace, cluster, user, reporter=None):
        job_meta = JobMeta(job_uuid, namespace, cluster, user)
        super(JobMetricCollector, self).__init__(job_meta, reporter=reporter)
        self._runtime_metric = RuntimeMetric([])
        self.dataset_metric = None
        self._flops = 0
        self._batch_size = 0
        self._report_runtime_thread = threading.Thread(
            target=self.report_runtime_stats_periodically, daemon=True
        )
        self._custom_metric: Dict[str, str] = {}

    @BaseMetricCollector.catch_exception
    def collect_dataset_metric(self, name, size, ds_type=DatasetType.TEXT):
        if not self.dataset_metric:
            self.dataset_metric = DatasetMetric.new_dataset_metric(
                ds_type, name, size
            )
        else:
            if not self.dataset_metric.name:
                self.dataset_metric.name = name
            if not self.dataset_metric.size:
                self.dataset_metric.size = size
        self._stats_reporter.report_dataset_metric(self.dataset_metric)

    @BaseMetricCollector.catch_exception
    def collect_training_hyper_params(self, epoch, batch_size):
        self._batch_size = batch_size
        params = TrainingHyperParams(batch_size, epoch)
        self._stats_reporter.report_training_hyper_params(params)

    @BaseMetricCollector.catch_exception
    def collect_job_type(self, job_type):
        self._stats_reporter.report_job_type(job_type)

    @BaseMetricCollector.catch_exception
    def collect_model_metric(
        self, tensor_stats: TensorStats, op_stats: OpStats
    ):
        if not self._flops:
            self._flops = op_stats.flops
        op_stats.flops = self._flops
        metric = ModelMetric(tensor_stats, op_stats)
        self._stats_reporter.report_model_metrics(metric)

    @BaseMetricCollector.catch_exception
    def _report_runtime_stats(self):
        self._stats_reporter.report_runtime_stats(self._runtime_metric)

    @BaseMetricCollector.catch_exception
    def collect_custom_data(self):
        self._stats_reporter.report_customized_data(self._custom_metric)

    def collect_runtime_stats(
        self, speed_monitor: SpeedMonitor, running_nodes: List[Node]
    ):
        """Set runtime info by global step"""
        if (
            speed_monitor.running_speed == 0
            or not speed_monitor.all_worker_joined()
        ):
            return
        self._runtime_metric.clear()
        self._runtime_metric.global_step = speed_monitor.completed_global_step
        self._runtime_metric.timestamp = int(time.time())
        self._runtime_metric.speed = speed_monitor.running_speed

        if (
            speed_monitor.init_training_time > 0
            and self._custom_metric.get(CustomMetricKey.INIT_TRAINING_TIME, 0)
            == 0
        ):
            self._custom_metric[
                CustomMetricKey.INIT_TRAINING_TIME
            ] = speed_monitor.init_training_time
            self.collect_custom_data()
        for node in running_nodes:
            node_sample = copy.deepcopy(node)
            node_sample.used_resource.memory *= MemoryUnit.MB
            if (node.type, node.id) in speed_monitor.running_workers:
                self._runtime_metric.running_nodes.append(node_sample)
            else:
                self._runtime_metric.running_nodes.append(node_sample)
        if not self._report_runtime_thread.is_alive():
            self._report_runtime_thread.start()

    def report_runtime_stats_periodically(self):
        global_step = 0
        while True:
            if (
                self._runtime_metric.global_step > global_step
                and self._runtime_metric.speed > 0
            ):
                self._report_runtime_stats()
                global_step = self._runtime_metric.global_step
            time.sleep(15)

    @BaseMetricCollector.catch_exception
    def collect_job_exit_reason(self, reason):
        self._stats_reporter.report_job_exit_reason(reason)
