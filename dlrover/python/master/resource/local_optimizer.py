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

from dlrover.python.master.resource.optimizer import (
    ResourceOptimizer,
    ResourcePlan,
)
from typing import Dict, List
from dlrover.python.common.node import Node, NodeResource, NodeGroupResource
from dlrover.python.common.constants import NodeType
from dlrover.python.master.stats.reporter import LocalStatsCollector


class OptimizerParams(object):
    def __init__(self):
        self.ps_cpu_overload_threshold = 0.9
        self.worker_memory_margin_percent = 0.4


class LocalOptimizer(ResourceOptimizer):
    name = "local"

    def __init__(self, job_uuid):
        self._job_uuid = job_uuid
        self._stats_collector = LocalStatsCollector()
        self._opt_params = OptimizerParams()

    def generate_opt_plan(self, stage, config={}):
        return ResourcePlan()

    def generate_oom_recovery_plan(self, oom_pods, stage, config={}):
        return ResourcePlan()

    def generate_job_create_resource(self):
        plan = ResourcePlan()
        node_cpu = 16
        if self._resource_limits.cpu < 32:
            node_cpu = 1
        node_memory = 16 * 1024
        if self._resource_limits.memory < 32:
            node_memory = 512

        ps = NodeGroupResource(1, NodeResource(node_cpu, node_memory))
        worker = NodeGroupResource(1, NodeResource(node_cpu, node_memory))
        plan.node_group_resources[NodeType.WORKER] = worker
        plan.node_group_resources[NodeType.PS] = ps
        return plan

    def generate_ps_initial_resource(self):
        resources = self._extract_node_runtime_resource()
        total_ps_cpus = []
        total_ps_memory = []
        for resources in resources[NodeType.PS]:
            cpu, memory = 0, 0
            for r in resources:
                cpu += r.cpu
                memory += r.memory
            total_ps_cpus.append(cpu)
            total_ps_memory.append(memory)
                
        plan = ResourcePlan()
        return plan

    def generate_worker_initial_resoruce(self):
        node_samples = self._extract_node_runtime_resource()
        max_ps_cpu_util = 0.0
        for nodes in node_samples[NodeType.PS]:
            for node in nodes:
                cpu_util = node.used_resource.cpu / node.config_resource.cpu
                max_ps_cpu_util = max(cpu_util, max_ps_cpu_util)

        opt_worker_num = len(node_samples[NodeType.WORKER])
        factor = self._opt_params.ps_cpu_overload_threshold / max_ps_cpu_util
        if factor > 1:
            opt_worker_num = int(opt_worker_num * factor)

        worker_cpus = 0.0
        worker_memory = 0.0
        for nodes in node_samples[NodeType.WORKER]:
            for node in nodes:
                worker_cpus.append(node.used_resource.cpu)
                worker_memory = max(node.used_resource.memory, worker_memory)
        opt_cpu = sum(worker_cpus) / len(worker_cpus)
        opt_memory = (1 + self._opt_params.worker_memory_margin_percent) * worker_memory

        ps_resource = self._compute_total_requested_resource(NodeType.PS)
        remaining_cpu = self._resource_limits.cpu - ps_resource.cpu
        remaining_memory = self._resource_limits.memory - ps_resource.memory
        max_worker_num = min(remaining_cpu / opt_cpu, remaining_memory / opt_memory)
        opt_worker_num = min(max_worker_num, opt_worker_num)
    
        plan = ResourcePlan()
        plan.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            opt_worker_num, NodeResource(opt_cpu, opt_memory)
        )
        return plan

    def generate_running_resource(self):
        plan = ResourcePlan()
        return plan

    def _extract_node_runtime_resource(self) -> Dict[str, List[List[Node]]]:
        stats = self._stats_collector.get_runtime_stats()
        node_used_resources: Dict[str, List[List[Node]]] = {}
        node_used_resources[NodeType.PS] = []
        node_used_resources[NodeType.WORKER] = []

        latest_ps = set()
        latest_worker_num = 0
        for node in stats[-1].running_pods:
            if node.type in [NodeType.CHIEF, NodeType.WORKER]:
                latest_worker_num += 1
            elif node.type == NodeType.PS:
                latest_ps.add(node.id)

        for stat in stats:
            cur_ps_resources = []
            cur_worker_resources = []
            cur_ps = set()
            cur_worker_num = 0
            for node in stat.running_pods:
                if node.type in [NodeType.CHIEF, NodeType.WORKER]:
                    cur_worker_resources.append(node)
                    cur_worker_num += 1
                elif node.type == NodeType.PS:
                    cur_ps_resources.append(node)
                    cur_ps.add(node.id)
            if cur_ps == latest_ps:
                node_used_resources[NodeType.PS].append(cur_ps_resources)
            if latest_worker_num == cur_worker_num:
                node_used_resources[NodeType.WORKER].append(
                    cur_worker_resources
                )
        return node_used_resources

    def _compute_total_requested_resource(self, type):
        stats = self._stats_collector.get_runtime_stats()
        cpu = 0
        memory = 0
        for node in stats[-1].running_pods:
            if type == type:
                cpu += node.config_resource.cpu
                memory += node.config_resource.memory
                
        return NodeResource(cpu, memory)
