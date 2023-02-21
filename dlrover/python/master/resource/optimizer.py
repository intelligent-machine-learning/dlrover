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

from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import (
    DefaultNodeResource,
    NodeResourceLimit,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.common.serialize import JsonSerializable
from dlrover.python.scheduler.job import ResourceLimits

_dlrover_context = Context.singleton_instance()


def _limit_cpu(cpu):
    if cpu > 0:
        if cpu < NodeResourceLimit.MIN_CPU_CORES:
            cpu = NodeResourceLimit.MIN_CPU_CORES
        elif cpu > NodeResourceLimit.MAX_CPU_CORES:
            cpu = NodeResourceLimit.MAX_CPU_CORES
    return cpu


def _limit_memory(memory):
    if memory > 0:
        if memory < NodeResourceLimit.MIN_MEMORY:
            memory = NodeResourceLimit.MIN_MEMORY
        elif memory > NodeResourceLimit.MAX_MEMORY:
            memory = NodeResourceLimit.MAX_MEMORY
    return memory


class ResourcePlan(JsonSerializable):
    """A resource configuration plan."""

    def __init__(self):
        self.node_group_resources: Dict[str, NodeGroupResource] = {}
        self.node_resources: Dict[str, NodeResource] = {}

    def empty(self):
        return len(self.node_group_resources) + len(self.node_resources) == 0

    def adjust_plan_by_context(self):
        if not _dlrover_context.auto_ps_enabled:
            if NodeType.PS in self.node_group_resources:
                del self.node_group_resources[NodeType.PS]
            self.node_resources.clear()

        if (
            not _dlrover_context.auto_worker_enabled
            and NodeType.WORKER in self.node_group_resources
        ):
            del self.node_group_resources[NodeType.WORKER]

    def limit_resource_value(self):
        for type, group in self.node_group_resources.items():
            resource = group.node_resource
            resource.cpu = _limit_cpu(resource.cpu)
            resource.memory = _limit_memory(resource.memory)

            if type == NodeType.WORKER:
                group.count = min(
                    group.count, NodeResourceLimit.MAX_WORKER_NUM
                )
                if group.count > 0:
                    resource.cpu = (
                        resource.cpu
                        if resource.cpu
                        else DefaultNodeResource.WORKER_CPU
                    )
                    resource.memory = (
                        resource.memory
                        if resource.memory
                        else DefaultNodeResource.WORKER_MEMORY
                    )
            elif type == NodeType.PS:
                group.count = min(group.count, NodeResourceLimit.MAX_PS_NUM)
                if group.count > 0:
                    resource.cpu = (
                        resource.cpu
                        if resource.cpu
                        else DefaultNodeResource.PS_CPU
                    )
                    resource.memory = (
                        resource.memory
                        if resource.memory
                        else DefaultNodeResource.PS_MEMORY
                    )

            for _, resource in self.node_resources.items():
                resource.cpu = _limit_cpu(resource.cpu)
                resource.memory = _limit_memory(resource.memory)

    @classmethod
    def new_default_plan(cls):
        plan = ResourcePlan()
        plan.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            DefaultNodeResource.WORKER_NUM,
            NodeResource(
                DefaultNodeResource.WORKER_CPU,
                DefaultNodeResource.WORKER_MEMORY,
            ),
        )
        plan.node_group_resources[NodeType.PS] = NodeGroupResource(
            DefaultNodeResource.PS_NUM,
            NodeResource(
                DefaultNodeResource.PS_CPU,
                DefaultNodeResource.PS_MEMORY,
            ),
        )
        return plan


class ResourceOptimizer(metaclass=ABCMeta):
    def __init__(self, job_uuid, resource_limits: ResourceLimits):
        self._job_uuid = job_uuid
        self._resource_limits = resource_limits

    def update_job_uuid(self, job_uuid):
        self._job_uuid = job_uuid

    @abstractmethod
    def generate_opt_plan(self, stage, config={}) -> ResourcePlan:
        """Generate a resource configuration plan"""
        pass

    @abstractmethod
    def generate_oom_recovery_plan(
        self, oom_nodes, stage, config={}
    ) -> ResourcePlan:
        """Generate a recovery plan for OOM nodes"""
        pass

    @abstractmethod
    def generate_resource_plan_with_optimizer(self, config={}) -> ResourcePlan:
        """Generate a resource plan by an optimizer"""
        pass


class SimpleOptimizer(ResourceOptimizer):
    def __init__(self, job_uuid, resource_limits: ResourceLimits):
        self._job_uuid = job_uuid
        self._resource_limits = resource_limits

    def update_job_uuid(self, job_uuid):
        self._job_uuid = job_uuid

    def generate_opt_plan(self, stage, config={}) -> ResourcePlan:
        """Generate a resource configuration plan"""
        return ResourcePlan()

    def generate_oom_recovery_plan(
        self, oom_nodes, stage, config={}
    ) -> ResourcePlan:
        """Generate a recovery plan for OOM nodes"""
        return ResourcePlan()

    def generate_resource_plan_with_optimizer(self, config={}) -> ResourcePlan:
        return ResourcePlan()
