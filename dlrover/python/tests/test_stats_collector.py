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
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.stats.reporter import JobMeta, LocalStatsReporter
from dlrover.python.master.stats.training_metrics import RuntimeMetric


class LocalStatsCollectorTest(unittest.TestCase):
    def test_report_resource_usage(self):
        job_meta = JobMeta("1111")
        reporter = LocalStatsReporter(job_meta)
        reporter._runtime_stats = []
        ps = Node(
            NodeType.PS,
            0,
            config_resource=NodeResource(4, 4096),
            name="ps-0",
        )
        worker = Node(
            NodeType.WORKER, 0, config_resource=NodeResource(6, 4096)
        )
        worker.used_resource = NodeResource(1, 2048)
        ps.used_resource = NodeResource(1.0, 2048)
        for i in range(3):
            nodes = [ps]
            for i in range(3):
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=12, timestamp=ts)
            reporter.report_runtime_stats(m)
        self.assertEqual(len(reporter._runtime_stats), 3)

        for i in range(10):
            nodes = [ps]
            for i in range(4):
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=12, timestamp=ts)
            reporter.report_runtime_stats(m)
        self.assertEqual(len(reporter._runtime_stats), 8)


class StatsCollectorTest(unittest.TestCase):
    def test_job_metric_collector(self):
        collector = JobMetricCollector("1111", "default", "local", "dlrover")
        collector.collect_dataset_metric("test", 1000)

        speed_monitor = SpeedMonitor()
        t = int(time.time())
        speed_monitor.set_target_worker_num(1)
        speed_monitor.collect_global_step(100, t)
        speed_monitor.collect_global_step(1100, t + 10)
        speed_monitor.add_running_worker(NodeType.WORKER, 0)
        worker = Node(NodeType.WORKER, 0, None)
        collector._stats_reporter._runtime_stats = []
        collector.collect_runtime_stats(speed_monitor, [worker])
        self.assertEqual(len(collector._runtime_metric.running_nodes), 1)
        self.assertEqual(collector._runtime_metric.speed, 100)
        self.assertEqual(len(collector._stats_reporter._runtime_stats), 1)
