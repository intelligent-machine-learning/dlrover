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

import time
import unittest

from dlrover.python.common.constants import NodeType
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.stats.reporter import JobMeta, LocalStatsReporter
from dlrover.python.master.stats.training_metrics import RuntimeMetric
from dlrover.python.master.watcher.base_watcher import Node


class LocalStatsCollectorTest(unittest.TestCase):
    def test_report_resource_usage(self):
        job_meta = JobMeta("1111")
        reporter = LocalStatsReporter(job_meta)
        reporter._runtime_stats = []
        reporter.report_runtime_stats(RuntimeMetric([]))
        reporter.report_runtime_stats(RuntimeMetric([]))
        self.assertEqual(len(reporter._runtime_stats), 2)


class StatsCollectorTest(unittest.TestCase):
    def test_job_metric_collector(self):
        collector = JobMetricCollector("1111", "default", "local", "dlrover")
        collector.collect_dataset_metric("test", 1000)

        speed_monitor = SpeedMonitor()
        t = int(time.time())
        speed_monitor.collect_global_step(100, t)
        speed_monitor.collect_global_step(1100, t + 10)
        speed_monitor.add_running_worker(NodeType.WORKER, 0)
        worker = Node(NodeType.WORKER, 0, None)
        collector._stats_reporter._runtime_stats = []
        collector.collect_runtime_stats(speed_monitor, [worker])
        self.assertEqual(len(collector._runtime_metric.running_nodes), 1)
        self.assertEqual(collector._runtime_metric.speed, 100)
        self.assertEqual(len(collector._stats_reporter._runtime_stats), 1)
