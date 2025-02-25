# Copyright 2024 The DLRover Authors. All rights reserved.
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
from collections import OrderedDict
from datetime import datetime
from typing import Dict

from dlrover.python.common.global_context import Context, DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.metric import XpuNodeMetric
from dlrover.python.common.singleton import Singleton

_dlrover_context = Context.singleton_instance()


class JobMetricContext(Singleton):
    """
    JobMetricContext includes metrics and events among all nodes
    that be shared across all components
    """

    def __init__(self):
        """
        job metrics dict is a dict with timestamp as key,
        and the value is another dict with worker node id as key,
        and xpu metric as value
        """
        self._xpu_job_metrics: OrderedDict[
            int, Dict[str, XpuNodeMetric]
        ] = OrderedDict()
        self.max_metric_records = DefaultValues.MAX_METRIC_REC
        self._lock = threading.Lock()

    def backtrace_avg_metrics(self, metric, depth):
        """
        backtrace all avg metrics of all nodes

        Args:
            metric: name, e.g. GPU_TENSOR_UTIL
            depth: maximum depth of backtrace

        Returns:
            OrderedDict with key as timestamp, value as avg metric

        """
        with self._lock:
            try:
                od = OrderedDict()
                for tm in list(self._xpu_job_metrics.keys())[::-1]:
                    total = 0
                    for v in self._xpu_job_metrics[tm].values():
                        total += v.get_avg_metric(metric)
                    od[tm] = round(
                        total / len(self._xpu_job_metrics[tm].values()), 2
                    )
                    depth = depth - 1
                    if depth <= 0:
                        break
                return od
            except Exception as e:
                logger.error(f"bt_avg_metrics {metric} error: {e}")
                return None

    def backtrace_node_metrics(self, metric, depth):
        """
        backtrace all node avg metric lists

        Args:
            metric: name, e.g. GPU_TENSOR_UTIL
            depth: maximum backtrace depth

        Returns:
            OrderedDict with key as timestamp, value as node metric list

        """
        with self._lock:
            try:
                od = OrderedDict()
                for tm in list(self._xpu_job_metrics.keys())[::-1]:
                    od[tm] = []
                    for v in self._xpu_job_metrics[tm].values():
                        od[tm].append(v.get_avg_metric(metric))
                    depth = depth - 1
                    if depth <= 0:
                        break
                return od
            except Exception as e:
                logger.error(f"bt_node_metrics {metric} error: {e}")
                return None

    def backtrace_rank_metrics(self, metric, depth):
        """
        backtrace a number of rank metric lists

        Args:
            metric: name, e.g. GPU_TENSOR_UTIL
            depth: maximum backtrace depth

        Returns:
            OrderedDict with key as timestamp, value as rank metric list

        """
        with self._lock:
            try:
                od = OrderedDict()
                for tm in list(self._xpu_job_metrics.keys())[::-1]:
                    od[tm] = []
                    for v in self._xpu_job_metrics[tm].values():
                        od[tm].append(v.get_node_metrics(metric))
                    depth = depth - 1
                    if depth <= 0:
                        break
                return od
            except Exception as e:
                logger.error(f"bt_rank_metrics {metric} error: {e}")
                return None

    def log_job_metric(self, metric):
        """
        print job avg metrics of type metric among all nodes
        """
        try:
            for tm, metrics in self._xpu_job_metrics.items():
                dt_obj = datetime.fromtimestamp(tm)
                dt_str = "{}-{}-{} {}:{}:{}".format(
                    dt_obj.year,
                    dt_obj.month,
                    dt_obj.day,
                    dt_obj.hour,
                    dt_obj.minute,
                    dt_obj.second,
                )
                total = 0
                for v in metrics.values():
                    total += v.get_avg_metric(metric)

                logger.info(
                    f"{metric}[{dt_str}]: "
                    f"{round(total/len(metrics.values()), 2)}"
                )

        except Exception as e:
            logger.error(f"log_job_metric error: {e}")

    def add_node_metrics(
        self, timestamp: int, metrics: Dict[str, XpuNodeMetric]
    ) -> None:
        with self._lock:
            keys = list(self._xpu_job_metrics.keys())
            if len(keys) > 0 and timestamp <= keys[-1]:
                # timestamp should be sorted
                return
            elif len(keys) >= self.max_metric_records:
                # remove first item
                self._xpu_job_metrics.popitem(last=False)
            self._xpu_job_metrics[timestamp] = metrics

    def clear_node_metrics(self) -> None:
        with self._lock:
            self._xpu_job_metrics.clear()

    def size(self):
        with self._lock:
            return len(self._xpu_job_metrics)

    def get_latest_node_metrics(self):
        with self._lock:
            keys = list(self._xpu_job_metrics.keys())
            if len(keys) == 0:
                return None
            key = keys[-1]
            return key, self._xpu_job_metrics[key].copy()

    def get_earliest_node_metrics(self):
        with self._lock:
            keys = list(self._xpu_job_metrics.keys())
            if len(keys) == 0:
                return None
            key = keys[0]
            return key, self._xpu_job_metrics[key].copy()

    def get_node_metrics(self):
        with self._lock:
            return self._xpu_job_metrics.copy()


def get_job_metric_context() -> JobMetricContext:
    job_metric_context = JobMetricContext.singleton_instance()
    return job_metric_context
