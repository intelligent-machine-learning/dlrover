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

import unittest

from dlrover.python.common.constants import JobOptStage, NodeType
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.resource.local_optimizer import PSLocalOptimizer
from dlrover.python.master.stats.training_metrics import RuntimeMetric
from dlrover.python.scheduler.job import ResourceLimits


class LocalOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        limits = ResourceLimits(10, 8192)
        self._optimizer = PSLocalOptimizer("1111", limits)
        for i in range(10):
            nodes = []
            ps = Node(
                NodeType.PS,
                0,
                config_resource=NodeResource(4, 4096),
                name="ps-0",
            )
            ps.used_resource = NodeResource(1.4, 2048)
            nodes.append(ps)

            ps = Node(
                NodeType.PS,
                1,
                config_resource=NodeResource(4, 4096),
                name="ps-1",
            )
            ps.used_resource = NodeResource(1.0, 2048)
            nodes.append(ps)

            for i in range(3):
                worker = Node(
                    NodeType.WORKER, i, config_resource=NodeResource(6, 4096)
                )
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=10, timestamp=ts)
            self._optimizer._stats_collector.report_runtime_stats(m)

    def test_generate_oom_recovery_plan(self):
        node = Node(NodeType.WORKER, 0, config_resource=NodeResource(4, 4096))
        node.used_resource = NodeResource(2.4, 2048)
        plan = self._optimizer.generate_oom_recovery_plan(
            [node], JobOptStage.RUNNING
        )
        self.assertEqual(plan.node_resources[node.name].memory, 8192)

    def test_generate_job_create_resource(self):
        plan = self._optimizer._generate_job_create_resource()
        worker = plan.node_group_resources[NodeType.WORKER]
        self.assertEqual(worker.count, 1)
        self.assertEqual(worker.node_resource.cpu, 2)
        self.assertEqual(worker.node_resource.memory, 1639)
        ps = plan.node_group_resources[NodeType.PS]
        self.assertEqual(ps.count, 1)
        self.assertEqual(ps.node_resource.cpu, 2)
        self.assertEqual(ps.node_resource.memory, 1639)

    def test_extract_node_resource(self):
        node_samples = self._optimizer._extract_node_resource()
        self.assertEqual(len(node_samples[NodeType.WORKER]), 5)
        latest_workers = node_samples[NodeType.WORKER][0]
        self.assertEqual(len(latest_workers), 3)
        latest_ps = node_samples[NodeType.PS][0]
        self.assertEqual(len(latest_ps), 2)

    def test_generate_ps_initial_resource(self):
        plan = self._optimizer._generate_ps_initial_resource()
        ps = plan.node_group_resources[NodeType.PS]
        self.assertEqual(ps.count, 2)
        self.assertEqual(ps.node_resource.cpu, 4)
        self.assertEqual(ps.node_resource.memory, int(2048 * 1.2))

        self._optimizer._resource_limits.cpu = 100
        plan = self._optimizer._generate_ps_initial_resource()
        ps = plan.node_group_resources[NodeType.PS]
        self.assertEqual(ps.count, 12)
        self.assertEqual(ps.node_resource.cpu, 4)
        self.assertEqual(ps.node_resource.memory, int(2048 * 1.2))

    def test_optimize_hot_ps_cpu(self):
        for i in range(10, 15):
            nodes = []
            ps = Node(
                NodeType.PS,
                0,
                config_resource=NodeResource(4, 4096),
                name="ps-0",
            )
            ps.used_resource = NodeResource(3.6, 2048)
            nodes.append(ps)

            ps = Node(
                NodeType.PS,
                1,
                config_resource=NodeResource(4, 4096),
                name="ps-1",
            )
            ps.used_resource = NodeResource(0.3, 2048)
            nodes.append(ps)

            for i in range(3):
                worker = Node(
                    NodeType.WORKER, i, config_resource=NodeResource(6, 4096)
                )
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=10, timestamp=ts)
            self._optimizer._stats_collector.report_runtime_stats(m)

        self._optimizer._resource_limits.cpu = 100
        plan = self._optimizer._optimize_hot_ps_cpu()
        self.assertEqual(len(plan.node_resources), 1)
        self.assertEqual(plan.node_resources["ps-0"].cpu, 32)

    def test_generate_worker_resoruce(self):
        for i in range(10, 15):
            nodes = []
            ps = Node(
                NodeType.PS,
                0,
                config_resource=NodeResource(4, 4096),
                name="ps-0",
            )
            ps.used_resource = NodeResource(1.0, 2048)
            nodes.append(ps)

            ps = Node(
                NodeType.PS,
                1,
                config_resource=NodeResource(4, 4096),
                name="ps-1",
            )
            ps.used_resource = NodeResource(0.3, 2048)
            nodes.append(ps)

            for i in range(3):
                worker = Node(
                    NodeType.WORKER, i, config_resource=NodeResource(6, 4096)
                )
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=10, timestamp=ts)
            self._optimizer._stats_collector.report_runtime_stats(m)
        self._optimizer._resource_limits.cpu = 100
        self._optimizer._resource_limits.memory = 102400
        plan = self._optimizer._generate_worker_resoruce()
        worker_resource = plan.node_group_resources[NodeType.WORKER]
        self.assertEqual(worker_resource.count, 10)
        self.assertEqual(worker_resource.node_resource.cpu, 1.0)
        self.assertEqual(worker_resource.node_resource.memory, 2048 * 1.5)

    def test_compute_worker_speed_ratio(self):
        for i in range(10, 15):
            nodes = []
            ps = Node(
                NodeType.PS,
                0,
                config_resource=NodeResource(4, 4096),
                name="ps-0",
            )
            ps.used_resource = NodeResource(1.0, 2048)
            nodes.append(ps)

            ps = Node(
                NodeType.PS,
                1,
                config_resource=NodeResource(4, 4096),
                name="ps-1",
            )
            ps.used_resource = NodeResource(0.3, 2048)
            nodes.append(ps)

            for i in range(3):
                worker = Node(
                    NodeType.WORKER, i, config_resource=NodeResource(6, 4096)
                )
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=12, timestamp=ts)
            self._optimizer._stats_collector.report_runtime_stats(m)
        ratio = self._optimizer._compute_worker_speed_ratio()
        self.assertEqual(ratio, 1)

        for i in range(15, 20):
            nodes = []
            ps = Node(
                NodeType.PS,
                0,
                config_resource=NodeResource(4, 4096),
                name="ps-0",
            )
            ps.used_resource = NodeResource(1.0, 2048)
            nodes.append(ps)

            ps = Node(
                NodeType.PS,
                1,
                config_resource=NodeResource(4, 4096),
                name="ps-1",
            )
            ps.used_resource = NodeResource(0.3, 2048)
            nodes.append(ps)

            for i in range(4):
                worker = Node(
                    NodeType.WORKER, i, config_resource=NodeResource(6, 4096)
                )
                worker.used_resource = NodeResource(1, 2048)
                nodes.append(worker)
            step = i * 100 + 1
            ts = i * 1000 + 1
            m = RuntimeMetric(nodes, global_step=step, speed=15, timestamp=ts)
            self._optimizer._stats_collector.report_runtime_stats(m)

        ratio = self._optimizer._compute_worker_speed_ratio()
        self.assertEqual(ratio, 0.75)
