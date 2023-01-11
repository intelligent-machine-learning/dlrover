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

import math
from typing import Dict, List

from dlrover.python.common.constants import JobOptStage, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.common.serialize import JsonSerializable
from dlrover.python.master.resource.optimizer import (
    ResourceOptimizer,
    ResourcePlan,
)
from dlrover.python.master.stats.reporter import JobMeta, LocalStatsReporter
from dlrover.python.master.stats.training_metrics import RuntimeMetric
from dlrover.python.scheduler.job import ResourceLimits

_LATEST_SAMPLE_COUNT = 5
_INITIAL_NODE_CPU = 16
_INITIAL_NODE_MEMORY = 16 * 1024  # 16Gi
_MINIKUBE_INITIAL_NODE_CPU = 1
_MINIKUBE_INITIAL_NODE_MEMORY = 512  # 512Mi


class OptimizerParams(object):
    def __init__(self):
        self.ps_cpu_overload_threshold = 0.9
        self.ps_cpu_hot_threshold = 0.9
        self.ps_memory_margin_percent = 0.2
        self.worker_memory_margin_percent = 0.5
        self.oom_memory_up_factor = 2
        self.node_max_cpu = 32


class ProcessResourceRequirement(JsonSerializable):
    def __init__(self, worker_cpu, ps_cpu, worker_memory) -> None:
        self.worker_cpu = worker_cpu
        self.ps_cpu = ps_cpu
        self.worker_memory = worker_memory


class LocalOptimizer(ResourceOptimizer):
    name = "local"

    def __init__(self, job_uuid, resource_limits: ResourceLimits):
        self._job_uuid = job_uuid
        self._stats_collector = LocalStatsReporter(JobMeta(job_uuid))
        self._opt_params = OptimizerParams()
        self._resource_limits = resource_limits

    def generate_opt_plan(self, stage, config={}):
        plan = ResourcePlan()
        if stage == JobOptStage.CREATE:
            plan = self._generate_job_create_resource()
        if stage == JobOptStage.WORKER_INITIAL:
            plan = self._generate_worker_resoruce()
        if stage == JobOptStage.PS_INITIAL:
            plan = self._generate_ps_initial_resource()
        if stage == JobOptStage.RUNNING:
            plan = self._generate_job_running_resource()
        if plan.empty():
            logger.info("Not support job stage %s", stage)
        else:
            logger.info("plan of stage %s is %s", stage, plan.toJSON(indent=4))
        return plan

    def generate_oom_recovery_plan(
        self, oom_nodes: List[Node], stage, config={}
    ):
        plan = ResourcePlan()
        for node in oom_nodes:
            factor = self._opt_params.oom_memory_up_factor
            opt_memory = factor * node.config_resource.memory
            plan.node_resources[node.name] = NodeResource(0, opt_memory)
        return plan

    def generate_resource_plan_with_optimizer(self, config={}) -> ResourcePlan:
        """Generate a resource plan by an optimizer"""
        pass

    def _generate_job_create_resource(self):
        plan = ResourcePlan()
        node_cpu = _INITIAL_NODE_CPU
        node_memory = _INITIAL_NODE_MEMORY
        if (
            self._resource_limits.cpu < 16
            and self._resource_limits.memory < 32 * 1024
        ):
            # Set a little resource to test an elastic job on minikube.
            node_cpu = _MINIKUBE_INITIAL_NODE_CPU
            node_memory = _MINIKUBE_INITIAL_NODE_MEMORY

        ps = NodeGroupResource(1, NodeResource(node_cpu, node_memory))
        worker = NodeGroupResource(1, NodeResource(node_cpu, node_memory))
        plan.node_group_resources[NodeType.WORKER] = worker
        plan.node_group_resources[NodeType.PS] = ps
        return plan

    def _generate_ps_initial_resource(self):
        node_samples = self._extract_node_resource()
        max_ps_memory = 0
        ps_cpu_requested = 0
        for node in node_samples[NodeType.PS][0]:
            max_ps_memory = max(max_ps_memory, node.used_resource.memory)
            ps_cpu_requested = max(node.config_resource.cpu, ps_cpu_requested)

        resource = self._estimate_process_require_resource()

        limit_cpu = self._resource_limits.cpu
        max_worker_num = limit_cpu / (resource.ps_cpu + resource.worker_cpu)
        opt_total_ps_cpu = limit_cpu - max_worker_num * resource.worker_cpu
        opt_ps_num = math.ceil(opt_total_ps_cpu / ps_cpu_requested)
        opt_ps_memory = int(
            max_ps_memory * (1 + self._opt_params.ps_memory_margin_percent)
        )
        plan = ResourcePlan()
        plan.node_group_resources[NodeType.PS] = NodeGroupResource(
            opt_ps_num, NodeResource(ps_cpu_requested, opt_ps_memory)
        )
        return plan

    def _generate_job_running_resource(self):
        plan = self._optimize_hot_ps_cpu()
        if plan:
            return plan
        plan = self._generate_worker_resoruce()
        return plan

    def _estimate_process_require_resource(self):
        node_samples = self._extract_node_resource()
        total_ps_cpus = []
        for nodes in node_samples[NodeType.PS]:
            cpu, memory = 0, 0
            for node in nodes:
                cpu += node.used_resource.cpu
                memory += node.used_resource.memory
            total_ps_cpus.append(cpu)
        avg_ps_cpu = sum(total_ps_cpus) / len(total_ps_cpus)

        worker_cpus = []
        worker_memory = 0.0
        for nodes in node_samples[NodeType.WORKER]:
            for node in nodes:
                worker_cpus.append(node.used_resource.cpu)
                worker_memory = max(worker_memory, node.used_resource.memory)
        worker_cpu = sum(worker_cpus) / len(worker_cpus)

        worker_num = 0
        stats: List[RuntimeMetric] = self._stats_collector.get_runtime_stats()
        for node in stats[-1].running_nodes:
            if node.type in [NodeType.CHIEF, NodeType.WORKER]:
                worker_num += 1
        ps_cpu_per_process = avg_ps_cpu / worker_num
        resource = ProcessResourceRequirement(
            worker_cpu, ps_cpu_per_process, worker_memory
        )
        logger.info("Training process needs %s", resource.toJSON())
        return resource

    def _generate_worker_resoruce(self):
        plan = ResourcePlan()
        node_samples = self._extract_node_resource()
        max_ps_cpu_util = 0.0
        for nodes in node_samples[NodeType.PS]:
            for node in nodes:
                cpu_util = node.used_resource.cpu / node.config_resource.cpu
                max_ps_cpu_util = max(cpu_util, max_ps_cpu_util)

        opt_worker_num = len(node_samples[NodeType.WORKER][0])
        if max_ps_cpu_util == 0:
            logger.warning("No CPU utilization of PS")
            return plan
        factor = self._opt_params.ps_cpu_overload_threshold / max_ps_cpu_util
        if factor > 1:
            opt_worker_num = int(opt_worker_num * factor)

        worker_cpus = []
        worker_memory = 0.0
        for nodes in node_samples[NodeType.WORKER]:
            for node in nodes:
                worker_cpus.append(node.used_resource.cpu)
                worker_memory = max(node.used_resource.memory, worker_memory)
        opt_cpu = round(sum(worker_cpus) / len(worker_cpus), 1)
        opt_memory = int(
            (1 + self._opt_params.worker_memory_margin_percent) * worker_memory
        )

        ps_resource = self._compute_total_requested_resource(NodeType.PS)
        remaining_cpu = self._resource_limits.cpu - ps_resource.cpu
        remaining_memory = self._resource_limits.memory - ps_resource.memory
        logger.info(
            "Remaining resource cpu : %s, memory %sMi",
            remaining_cpu,
            remaining_memory,
        )
        max_worker_num = min(
            remaining_cpu / opt_cpu, remaining_memory / opt_memory
        )
        opt_worker_num = int(min(max_worker_num, opt_worker_num))

        plan.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            opt_worker_num, NodeResource(opt_cpu, opt_memory)
        )
        return plan

    def _optimize_hot_ps_cpu(self):
        node_samples = self._extract_node_resource()
        ps_used_cpus: Dict[int, List[float]] = {}
        ps_config_cpu: Dict[int, float] = {}
        for nodes in node_samples[NodeType.PS]:
            for node in nodes:
                ps_used_cpus.setdefault(node.id, [])
                ps_used_cpus[node.id].append(node.used_resource.cpu)
                ps_config_cpu[node.id] = node.config_resource.cpu
        ps_avg_cpus: Dict[int, float] = {}
        hot_ps = []
        for ps_id, cpu in ps_config_cpu.items():
            avg_cpu = sum(ps_used_cpus[ps_id]) / len(ps_used_cpus[ps_id])
            ps_avg_cpus[ps_id] = avg_cpu
            if avg_cpu / cpu >= self._opt_params.ps_cpu_hot_threshold:
                hot_ps.append(ps_id)
        if len(hot_ps) == 0:
            return

        cur_worker_num = len(node_samples[NodeType.WORKER][0])
        resource = self._estimate_process_require_resource()
        process_cpu = resource.ps_cpu + resource.worker_cpu
        max_worker_num = int(self._resource_limits.cpu / process_cpu)
        tune_factor = max(1, max_worker_num / cur_worker_num)
        plan = ResourcePlan()
        for ps in hot_ps:
            tune_factor = min(
                self._opt_params.node_max_cpu / ps_avg_cpus[ps], tune_factor
            )
        for node in node_samples[NodeType.PS][0]:
            opt_cpu = round(ps_avg_cpus[node.id] * tune_factor, 1)
            if node.config_resource.cpu >= opt_cpu:
                continue
            plan.node_resources[node.name] = NodeResource(opt_cpu, 0.0)
        return plan

    def _extract_node_resource(self) -> Dict[str, List[List[Node]]]:
        stats = self._stats_collector.get_runtime_stats()
        node_used_resources: Dict[str, List[List[Node]]] = {}
        node_used_resources[NodeType.PS] = []
        node_used_resources[NodeType.WORKER] = []

        if len(stats) == 0:
            return node_used_resources

        latest_ps = set()
        latest_worker_num = 0
        for node in stats[-1].running_nodes:
            if node.type in [NodeType.CHIEF, NodeType.WORKER]:
                latest_worker_num += 1
            elif node.type == NodeType.PS:
                latest_ps.add(node.id)

        sample_index = max(0, len(stats) - _LATEST_SAMPLE_COUNT)
        for stat in reversed(stats[sample_index:]):
            cur_ps_samples = []
            cur_worker_samples = []
            cur_ps = set()
            cur_worker_num = 0
            for node in stat.running_nodes:
                if node.type in [NodeType.CHIEF, NodeType.WORKER]:
                    cur_worker_samples.append(node)
                    cur_worker_num += 1
                elif node.type == NodeType.PS:
                    cur_ps_samples.append(node)
                    cur_ps.add(node.id)
            if cur_ps == latest_ps:
                node_used_resources[NodeType.PS].append(cur_ps_samples)
            if latest_worker_num == cur_worker_num:
                node_used_resources[NodeType.WORKER].append(cur_worker_samples)

        for ps_samples in node_used_resources[NodeType.PS]:
            ps_resource = []
            for ps in ps_samples:
                ps_resource.append(
                    (ps.used_resource.cpu, ps.used_resource.memory)
                )
            logger.info("PS resource samples = %s", ps_resource)

        for ps_samples in node_used_resources[NodeType.WORKER]:
            ps_resource = []
            for ps in ps_samples:
                ps_resource.append(
                    (ps.used_resource.cpu, ps.used_resource.memory)
                )
            logger.info("worker resource samples = %s", ps_resource)
        return node_used_resources

    def _compute_total_requested_resource(self, type):
        stats: List[RuntimeMetric] = self._stats_collector.get_runtime_stats()
        cpu = 0
        memory = 0
        for node in stats[-1].running_nodes:
            if node.type == type:
                cpu += node.config_resource.cpu
                memory += node.config_resource.memory

        return NodeResource(cpu, memory)
