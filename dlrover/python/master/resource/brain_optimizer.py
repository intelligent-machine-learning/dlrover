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

import grpc

from dlrover.python.brain.client import GlobalBrainClient
from dlrover.python.common.constants import MemoryUnit
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.resource.optimizer import (
    ResourceOptimizer,
    ResourcePlan,
)

_BASE_CONFIG_RETRIEVER = "base_config_retriever"


def catch_brain_optimization_exception(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except grpc.RpcError as e:
            logger.warning("Fail to execute %s by %s", func.__name__, e)
            return ResourcePlan()

    return wrapper


def convert_plan_msg(plan_msg):
    """Convert a GRPC plan message to a ResourcePlan.
    Args:
        plan_msg: brain_pb2.JobOptimizePlan instance.
    """
    plan = ResourcePlan()
    if not plan_msg:
        return plan
    for type, group in plan_msg.resource.task_group_resources.items():
        count = group.count
        memory = int(group.resource.memory / MemoryUnit.MB)  # MiB
        cpu = math.ceil(group.resource.cpu)
        plan.node_group_resources[type] = NodeGroupResource(
            count, NodeResource(cpu, memory)
        )

    for name, resource in plan_msg.resource.pod_resources.items():
        cpu = int(resource.cpu)
        memory = int(resource.memory / MemoryUnit.MB)
        plan.node_resources[name] = NodeResource(cpu, memory)
    return plan


class BrainResoureOptimizer(ResourceOptimizer):
    """Query resource plan from the brain service."""

    def __init__(self, job_uuid, resource_limits):
        super(BrainResoureOptimizer, self).__init__(job_uuid, resource_limits)
        self._brain_client = GlobalBrainClient.BRAIN_CLIENT

    @catch_brain_optimization_exception
    def generate_opt_plan(self, stage, config={}) -> ResourcePlan:
        res = self._brain_client.get_optimization_plan(
            self._job_uuid,
            stage,
            _BASE_CONFIG_RETRIEVER,
            config,
        )
        if not res.job_optimize_plans:
            logger.info("No any optimization plan for PS")
            return ResourcePlan()

        plan_msg = res.job_optimize_plans[0]
        logger.info(
            "The optimization plan of %s with config %s is %s",
            stage,
            config,
            plan_msg,
        )
        plan = convert_plan_msg(plan_msg)
        return plan

    @catch_brain_optimization_exception
    def generate_oom_recovery_plan(self, oom_nodes, stage, config={}):
        res = self._brain_client.get_oom_resource_plan(
            oom_nodes,
            self._job_uuid,
            stage,
            _BASE_CONFIG_RETRIEVER,
            config,
        )
        if not res.job_optimize_plans:
            logger.info("No any optimization plan for PS")
            return

        plan_msg = res.job_optimize_plans[0]
        logger.info("The optimization plan of %s is %s", stage, plan_msg)
        plan = convert_plan_msg(plan_msg)
        return plan

    @catch_brain_optimization_exception
    def generate_resource_plan_with_optimizer(self, config):
        res = self._brain_client.get_optimizer_resource_plan(
            self._job_uuid,
            _BASE_CONFIG_RETRIEVER,
            config,
        )
        if not res.job_optimize_plans:
            logger.info("No any plan with %s", config)
            return

        plan = res.job_optimize_plans[0]
        logger.info("The resource plan of %s is %s", config, plan)
        return plan
