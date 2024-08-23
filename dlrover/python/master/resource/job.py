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

import copy
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.brain.client import GlobalBrainClient
from dlrover.python.common.constants import (
    JobOptStage,
    NodeResourceLimit,
    NodeType,
    OptimizeMode,
    OptimizeWorkerPhase,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.common.serialize import JsonSerializable
from dlrover.python.master.resource.brain_optimizer import (
    BrainResoureOptimizer,
)
from dlrover.python.master.resource.local_optimizer import PSLocalOptimizer
from dlrover.python.master.resource.optimizer import (
    ResourcePlan,
    SimpleOptimizer,
)
from dlrover.python.scheduler.job import ResourceLimits

_WORKER_OPTIMIZE_PHASE = "optimizer.worker.optimize-phase"

_dlrover_context = Context.singleton_instance()


def new_ps_resource_optimizer(
    optimize_mode: str, job_uuid, resource_limits: ResourceLimits
):
    logger.info(
        "New %s resource optimizer for job %s", optimize_mode, job_uuid
    )
    if optimize_mode == OptimizeMode.CLUSTER:
        if GlobalBrainClient.BRAIN_CLIENT.available():
            return BrainResoureOptimizer(job_uuid, resource_limits)
        else:
            logger.warning(
                "Brain service is not available, use a local optimizer"
            )
            return PSLocalOptimizer(job_uuid, resource_limits)
    elif optimize_mode == OptimizeMode.SINGLE_JOB:
        return PSLocalOptimizer(job_uuid, resource_limits)
    else:
        logger.warning(
            "Not support optimization mode %s, use a simple optimizer",
            optimize_mode,
        )
        return SimpleOptimizer(job_uuid, resource_limits)


class JobResource(JsonSerializable):
    def __init__(self):
        self.node_group_resources: Dict[str, NodeGroupResource] = {}

    def get_node_group_resource(self, node_type):
        return self.node_group_resources.get(node_type, None)

    def _get_group_node_num(self, node_type):
        if node_type in self.node_group_resources:
            return self.node_group_resources[node_type].count
        return 0

    def get_node_types(self):
        return list(self.node_group_resources.keys())

    def update_node_group_resource(self, node_type, num, cpu, memory):
        self.node_group_resources.setdefault(
            node_type,
            NodeGroupResource(
                count=0,
                node_resource=NodeResource(0, 0),
            ),
        )
        resource = self.node_group_resources[node_type]
        resource.count = num or resource.count
        resource.node_resource.cpu = cpu or resource.node_resource.cpu
        resource.node_resource.memory = memory or resource.node_resource.memory

    @property
    def worker_num(self):
        return self._get_group_node_num(NodeType.WORKER)

    @property
    def ps_num(self):
        return self._get_group_node_num(NodeType.PS)

    @property
    def evaluator_num(self):
        return self._get_group_node_num(NodeType.EVALUATOR)

    @property
    def chief_num(self):
        return self._get_group_node_num(NodeType.CHIEF)

    def init_job_node_meta(
        self,
        relaunch_on_worker_failure,
        service_create_fn,
        new_node_name_fn,
    ):
        """
        relaunch_on_worker_failure: int, the number of relaunches.
        service_create_fn: a callable function to create the name for a sevice.
        new_node_name_fn: a callable function to create the name for a node.
        return: a dict with pod_type as key, and another dict as value.
                The other dict uses pod id as key, and PodInfo as value.
        """
        job_nodes: Dict[str, Dict[int, Node]] = {}
        for node_type in self.get_node_types():
            group_resource = self.get_node_group_resource(node_type)
            config_resource = group_resource.node_resource
            group_nodes: Dict[int, Node] = {}
            for i in range(group_resource.count):
                group_nodes[i] = Node(
                    node_type=node_type,
                    node_id=i,
                    rank_index=i,
                    name=new_node_name_fn(node_type, i),
                    config_resource=copy.deepcopy(config_resource),
                    max_relaunch_count=relaunch_on_worker_failure,
                    service_addr=service_create_fn(node_type, i),
                )
            job_nodes[node_type] = group_nodes
        logger.info(
            "after initializing job node meta job_nodes are %s" % job_nodes
        )
        return job_nodes

    def adjust_worker_for_estimator(self):
        if (
            NodeType.CHIEF in self.node_group_resources
            and self.node_group_resources[NodeType.CHIEF].count > 0
        ) or (NodeType.WORKER not in self.node_group_resources):
            return

        worker = self.node_group_resources[NodeType.WORKER]
        if worker.count <= 0:
            return

        chief = self.node_group_resources.get(
            NodeType.CHIEF, NodeGroupResource.new_empty()
        )
        chief.count = 1
        chief.node_resource.cpu = worker.node_resource.cpu
        chief.node_resource.memory = worker.node_resource.memory
        self.node_group_resources[NodeType.CHIEF] = chief
        worker.count -= 1
        logger.info("self = %s", self.to_json())


class JobResourceOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def update_job_uuid(self, job_uuid):
        pass

    @abstractmethod
    def init_job_resource(self, job_resource: JobResource):
        """Initialize resource configuration for a job."""
        pass

    @abstractmethod
    def get_job_resource_plan(self) -> ResourcePlan:
        """Get resource plan for a job."""
        pass

    @abstractmethod
    def adjust_oom_resource(self, node: Node):
        """Adjust the resource configuration for OOM nodes"""
        pass

    @abstractmethod
    def get_config_resource(self) -> JobResource:
        pass


class PSJobResourceOptimizer(JobResourceOptimizer):
    """It generates resource configuration for a PS job."""

    def __init__(
        self,
        worker_resource: NodeGroupResource,
        ps_resource: NodeGroupResource,
        optimize_mode: str,
        job_uuid="",
        resource_limits=ResourceLimits(),
    ):
        self._worker_resource = worker_resource
        self._ps_resource = ps_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._original_ps_resource = copy.deepcopy(self._ps_resource)
        self._resource_optimizer = new_ps_resource_optimizer(
            optimize_mode, job_uuid, resource_limits
        )
        self._lock = threading.Lock()
        self.optimized_ps_mem = False
        self.optimize_worker_sampled = False
        self._job_stage = JobOptStage.CREATE
        self._last_ps_change_time = 0.0

    def set_job_stage(self, stage):
        self._job_stage = stage

    def get_job_stage(self):
        return self._job_stage

    def get_config_resource(self):
        job_config = JobResource()
        worker_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.WORKER] = worker_config
        ps_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.PS] = ps_config
        return job_config

    def update_job_uuid(self, job_uuid):
        self._resource_optimizer.update_job_uuid(job_uuid)

    def _init_job_resource_by_optimizer(self):
        plan = self._resource_optimizer.generate_opt_plan(self._job_stage)
        if not plan or plan.empty():
            logger.info("Use the default plan to start the job")
            plan = self._gen_default_resource_plan()
        self._job_stage = JobOptStage.WORKER_INITIAL

        if (
            _dlrover_context.auto_worker_enabled
            and NodeType.WORKER in plan.node_group_resources
        ):
            worker_resource = self._check_ignore_original_worker_resource(
                plan.node_group_resources[NodeType.WORKER]
            )
            self._worker_resource.update(
                worker_resource.count,
                worker_resource.node_resource.cpu,
                worker_resource.node_resource.memory,
            )
        if (
            _dlrover_context.auto_ps_enabled
            and NodeType.PS in plan.node_group_resources
        ):
            ps_resource = self._check_ignore_original_ps_resource(
                plan.node_group_resources[NodeType.PS]
            )
            self._ps_resource.update(
                ps_resource.count,
                ps_resource.node_resource.cpu,
                ps_resource.node_resource.memory,
            )

    def _gen_default_resource_plan(self):
        plan = ResourcePlan.new_default_plan()
        return plan

    def init_job_resource(self, job_resource: JobResource):
        """Adjust the initial resource of typed pods by EasyDL.
        Args:
            job_resource: node resource configuration of a job.
        """
        self._init_job_resource_by_optimizer()
        job_resource.update_node_group_resource(
            NodeType.WORKER,
            self._worker_resource.count,
            self._worker_resource.node_resource.cpu,
            self._worker_resource.node_resource.memory,
        )

        job_resource.update_node_group_resource(
            NodeType.PS,
            self._ps_resource.count,
            self._ps_resource.node_resource.cpu,
            self._ps_resource.node_resource.memory,
        )

        evaluator_group = job_resource.get_node_group_resource(
            NodeType.EVALUATOR
        )
        if evaluator_group:
            resource = evaluator_group.node_resource
            if resource.cpu < NodeResourceLimit.MIN_VALID_CPU:
                resource.cpu = self._worker_resource.node_resource.cpu
            min_memory = NodeResourceLimit.MIN_VALID_MEMORY
            if resource.memory < min_memory:
                resource.memory = self._worker_resource.node_resource.memory

        logger.info("Job resource = %s", job_resource.to_json())
        return job_resource

    def adjust_oom_resource(self, node):
        if node.type == NodeType.PS:
            self._adjust_oom_ps_resource(node)
        else:
            self._adjust_oom_worker_resource(node)

    def _adjust_oom_worker_resource(self, node: Node):
        """Increment the memory to launch worker. The new memory
        is max(1.5 * memory, the memory set by users).

        Args:
            node: Node object.
        """
        cur_mem = node.config_resource.memory
        if (
            _dlrover_context.auto_worker_enabled
            and self._job_stage == JobOptStage.WORKER_INITIAL
        ):
            plan = self._resource_optimizer.generate_oom_recovery_plan(
                [node], JobOptStage.CREATE
            )
            if plan and not plan.empty():
                new_resource = plan.node_group_resources[NodeType.WORKER]
                self._worker_resource.node_resource.memory = max(
                    self._worker_resource.node_resource.memory,
                    new_resource.node_resource.memory,
                )
        else:
            plan = self._get_worker_resource_at_init_phase()
            if NodeType.WORKER in plan.node_group_resources:
                new_resource = self._check_ignore_original_worker_resource(
                    plan.node_group_resources[NodeType.WORKER]
                )
                self._worker_resource.node_resource.memory = max(
                    self._worker_resource.node_resource.memory,
                    new_resource.node_resource.memory,
                )
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        cur_mem = min(cur_mem, NodeResourceLimit.MAX_MEMORY)
        opt_memory = int(
            max(
                self._worker_resource.node_resource.memory,
                cur_mem,
                self._original_worker_resource.node_resource.memory,
            )
        )
        incre_memory = opt_memory - node.config_resource.memory
        incre_memory = min(
            incre_memory, NodeResourceLimit.MAX_INCREMENTAL_MEMORY
        )
        node.config_resource.memory += incre_memory
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )

    def _adjust_oom_ps_resource(self, node: Node):
        """Adjust PS resource if there is a OOM PS"""
        plan = self._resource_optimizer.generate_oom_recovery_plan(
            [node], JobOptStage.PS_INITIAL
        )
        if plan and not plan.empty() and node.name in plan.node_resources:
            resource = plan.node_resources[node.name]
            self._ps_resource.node_resource.memory = max(
                self._ps_resource.node_resource.memory,
                resource.memory,
            )
        cur_mem = node.config_resource.memory
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        opt_memory = int(
            max(
                self._ps_resource.node_resource.memory,
                cur_mem,
                self._original_ps_resource.node_resource.memory,
            )
        )
        incre_memory = opt_memory - node.config_resource.memory
        incre_memory = min(
            incre_memory, NodeResourceLimit.MAX_INCREMENTAL_MEMORY
        )
        node.config_resource.memory += incre_memory
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )
        self._last_ps_change_time = time.time()

    def get_job_resource_plan(self):
        plan = None
        if self._job_stage == JobOptStage.WORKER_INITIAL:
            plan = self._get_worker_resource_at_init_phase()
            self._job_stage = JobOptStage.PS_INITIAL
        elif self._job_stage == JobOptStage.PS_INITIAL:
            plan = self._get_ps_resource_plan()
            self._job_stage = JobOptStage.PS_RUNNING
        elif self._job_stage == JobOptStage.PS_RUNNING:
            plan = self._get_ps_resource_plan()
            if plan.empty():
                plan = self._get_worker_resource_at_running()
        if not plan or plan.empty():
            return None

        if NodeType.WORKER in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.WORKER)

        if plan and NodeType.PS in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.PS)

        plan.adjust_plan_by_context()
        return plan

    def _get_worker_resource_at_running(self):
        if not self.optimize_worker_sampled:
            plan = self._get_worker_resource_at_sample_phase()
            self.optimize_worker_sampled = True
        else:
            plan = self._get_worker_resource_at_stable_phase()
        return plan

    def _get_worker_resource_at_init_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.INITIAL
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if plan.empty():
            logger.info("No any plan to initialize the number of worker")
        return plan

    def _get_worker_resource_at_sample_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.SAMPLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan or plan.empty():
            return
        return plan

    def _get_worker_resource_at_stable_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.STABLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_RUNNING, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_ps_resource_plan(self, optimizer_config={}):
        # The interval of changing PS should be long enough.
        interval = _dlrover_context.seconds_interval_to_change_ps
        if time.time() - self._last_ps_change_time > interval:
            plan = self._resource_optimizer.generate_opt_plan(
                self._job_stage, optimizer_config
            )
        else:
            logger.info(
                "Skip optimizing PS, because the interval"
                "to change ps is too short."
            )
            return ResourcePlan()
        if not plan.empty():
            self._last_ps_change_time = time.time()
        return plan

    def _verify_optimized_group_resource(self, plan: ResourcePlan, node_type):
        group = plan.node_group_resources[node_type]
        if node_type == NodeType.WORKER:
            group = self._check_ignore_original_worker_resource(group)
            node_resource = group.node_resource
            self._worker_resource.count = group.count
            self._worker_resource.node_resource.cpu = node_resource.cpu
            self._worker_resource.node_resource.memory = node_resource.memory
        elif node_type == NodeType.PS:
            group = self._check_ignore_original_ps_resource(group)
            node_resource = group.node_resource
            self._ps_resource.count = min(
                group.count, NodeResourceLimit.MAX_PS_NUM
            )
            self._ps_resource.node_resource.cpu = node_resource.cpu
            self._ps_resource.node_resource.memory = node_resource.memory
        return group

    def _check_ignore_original_worker_resource(
        self, resource: NodeGroupResource
    ):
        """Abandon the optimization result if users have set the resource."""
        #  Users may worry about that the increasing number of worker hurts the
        #  accuracy, so the max number of worker is the configuration.
        original_resource = self._original_worker_resource.node_resource
        if self._original_worker_resource.count > 0:
            resource.count = self._original_worker_resource.count
        if resource.node_resource.cpu == 0:
            resource.node_resource.cpu = original_resource.cpu
        if resource.node_resource.memory == 0:
            resource.node_resource.memory = original_resource.memory
        return resource

    def _check_ignore_original_ps_resource(self, resource: NodeGroupResource):
        """Abandon the optimization result if users have set the resource."""
        original_resource = self._original_ps_resource.node_resource
        if self._original_ps_resource.count > 0:
            resource.count = self._original_ps_resource.count
        if original_resource.memory >= NodeResourceLimit.MIN_VALID_MEMORY:
            resource.node_resource.memory = original_resource.memory
        if original_resource.cpu >= NodeResourceLimit.MIN_VALID_CPU:
            resource.node_resource.cpu = original_resource.cpu
        return resource


class AllreduceJobResourceOptimizer(JobResourceOptimizer):
    """It generates resource configuration for a job."""

    def __init__(
        self,
        worker_resource: NodeGroupResource,
        job_uuid="",
    ):
        self._worker_resource = worker_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._job_uuid = job_uuid
        self._lock = threading.Lock()
        self._node_unit = 1
        self._alive_node_num = 0

    def update_job_uuid(self, job_uuid):
        pass

    def init_job_resource(self, job_resource: JobResource):
        pass

    def get_job_resource_plan(self) -> ResourcePlan:
        """Check wether there are free nodes in the cluster."""
        plan = ResourcePlan()
        worker_config = copy.deepcopy(self._original_worker_resource)
        max_node_num = self._original_worker_resource.count
        request_num = max_node_num - self._alive_node_num
        free_num = self._get_free_gpu_node()
        free_num = (free_num // self._node_unit) * self._node_unit
        new_num = min(free_num, request_num)
        worker_config.count = self._alive_node_num + new_num
        plan.node_group_resources[NodeType.WORKER] = worker_config
        return plan

    # TODO: implement the function to query the number free GPU nodes.
    def _get_free_gpu_node(self):
        return 0

    def adjust_oom_resource(self, node: Node):
        """Adjust the resource configuration for OOM nodes"""
        node.config_resource.memory *= 2

    def get_config_resource(self):
        job_config = JobResource()
        worker_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.WORKER] = worker_config
        return job_config

    def set_node_unit(self, node_unit):
        self._node_unit = node_unit

    def set_alive_node_num(self, node_num):
        self._alive_node_num = node_num
