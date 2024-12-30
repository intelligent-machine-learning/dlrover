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
from typing import Dict

from dlrover.python.common.global_context import Context
from dlrover.python.common.metric.metric import XpuNodeMetric
from dlrover.python.common.singleton import Singleton

_dlrover_context = Context.singleton_instance()


class JobMetricContext(Singleton):
    """
    JobMetricContext includes metrics and events among all nodes
    that be shared across all components
    """

    def __init__(self):
        self._lock = threading.Lock()
        """
        job metrics dict is a dict with timestamp as key,
        and the value is another dict with worker node id as key,
        and xpu metric as value
        """
        self._xpu_job_metrics: OrderedDict[
            int, Dict[str, XpuNodeMetric]
        ] = OrderedDict()
        self.max_metric_records = _dlrover_context.max_metric_records

    def add_node_metrics(
        self, timestamp: int, metrics: Dict[str, XpuNodeMetric]
    ) -> None:
        with self._lock:
            keys = list(self._xpu_job_metrics.keys())
            if len(keys) > 0 and timestamp <= keys[-1]:
                """timestamp should be sorted"""
                return
            elif len(keys) >= self.max_metric_records:
                """remove first item"""
                self._xpu_job_metrics.popitem(last=False)
            self._xpu_job_metrics[timestamp] = metrics

    def clear_node_metrics(self) -> None:
        with self._lock:
            self._xpu_job_metrics = OrderedDict()

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
