# Copyright 2021 The ElasticDL Authors. All rights reserved.
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
import math
import threading

from dlrover.proto import brain_pb2
from dlrover.python.common.global_context import Context
from dlrover.python.common.constants import (
    NodeType,
    NodeResourceLimit,
    DefaultNodeResource,
    JobOptStage,
    OptimizeWorkerPhase,
)
from dlrover.python.common.log_utils import default_logger as logger

_BASE_CONFIG_RETRIEVER = "base_config_retriever"

_WORKER_OPTIMIZE_PHASE = "optimizer.worker.optimize-phase"


_dlrover_context = Context.instance()


def get_default_plan():
    plan = brain_pb2.JobOptimizePlan()
    memory_byte = 1024 * 1024
    group_resources = plan.resource.task_group_resources
    group_resources["worker"].count = DefaultNodeResource.WORKER_NUM
    worker_memory = DefaultNodeResource.WORKER_MEMORY * memory_byte
    group_resources["worker"].resource.memory = worker_memory
    group_resources["worker"].resource.cpu = DefaultNodeResource.WORKER_CPU

    group_resources["ps"].count = DefaultNodeResource.PS_NUM
    ps_memory = DefaultNodeResource.PS_MEMORY * memory_byte
    group_resources["ps"].resource.memory = ps_memory
    group_resources["ps"].resource.cpu = DefaultNodeResource.PS_CPU
    return plan


def extract_resource_from_plan(plan, pod_type):
    group_resource = plan.resource.task_group_resources[pod_type]
    resource = group_resource.resource
    count = group_resource.count
    memory = int(resource.memory / 1024.0 / 1024.0)  # MiB
    cpu = math.ceil(resource.cpu)
    cpu, memory = _check_resource_limits(cpu, memory)
    if pod_type == NodeType.WORKER:
        count = min(count, NodeResourceLimit.MAX_WORKER_NUM)
        if count > 0:
            cpu = cpu if cpu else DefaultNodeResource.WORKER_CPU
            memory = memory if memory else DefaultNodeResource.WORKER_MEMORY
    elif pod_type == NodeType.PS:
        count = min(count, NodeResourceLimit.MAX_PS_NUM)
        if count > 0:
            cpu = cpu if cpu else DefaultNodeResource.PS_CPU
            memory = memory if memory else DefaultNodeResource.PS_MEMORY
    return count, cpu, memory


def adjust_plan_by_context(plan):
    if not _dlrover_context.easydl_ps_enabled:
        if NodeType.PS in plan.resource.task_group_resources:
            del plan.resource.task_group_resources[NodeType.PS]
        plan.resource.pod_resources.clear()

    if (
        not _dlrover_context.easydl_worker_enabled
        and NodeType.WORKER in plan.resource.task_group_resources
    ):
        del plan.resource.task_group_resources[NodeType.WORKER]

    for _, resource in plan.resource.pod_resources.items():
        memory = int(resource.memory / 1024.0 / 1024.0)  # MiB
        cpu, memory = _check_resource_limits(resource.cpu, memory)
        resource.cpu = cpu
        resource.memory = memory

    return plan


def _check_resource_limits(cpu, mem):
    if cpu > 0:
        if cpu < NodeResourceLimit.MIN_CPU_CORES:
            cpu = NodeResourceLimit.MIN_CPU_CORES
        elif cpu > NodeResourceLimit.MAX_CPU_CORES:
            cpu = NodeResourceLimit.MAX_CPU_CORES

    if mem > 0:
        if mem < NodeResourceLimit.MIN_MEMORY:
            mem = NodeResourceLimit.MIN_MEMORY
        elif mem > NodeResourceLimit.MAX_MEMORY:
            mem = NodeResourceLimit.MAX_MEMORY
    return cpu, mem


class TaskGroupResource(object):
    def __init__(self, num, cpu, mem):
        self.num = num
        self.cpu = cpu
        self.mem = mem  # Mi

    def update(self, num=0, cpu=0, mem=0):
        self.num = num if num > 0 else self.num
        self.cpu = cpu if cpu > 0 else self.cpu
        self.mem = mem if mem > 0 else self.mem


def check_resource_plan_empty(plan):
    if not plan or not plan.resource:
        return True
    if (
        len(plan.resource.task_group_resources) == 0
        and len(plan.resource.pod_resources) == 0
    ):
        return True


class JobResourceScaler(object):
    """The pod resource scaler will propose the pod resource configuration
    including CPU and memory. For example, it will query the EasyDL server
    to get the used CPU and memory of jobs which is similar to the job.
    """

    def __init__(
        self,
        job_name,
        job_uuid,
        worker_resource,
        ps_resource,
        k8s_client=None,
        k8s_cluster=None,
    ):
        self._job_name = job_name
        self._job_uuid = job_uuid
        self._k8s_cluster = k8s_cluster
        self._worker_resource = worker_resource
        self._ps_resource = ps_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._original_ps_resource = copy.deepcopy(self._ps_resource)
        self._k8s_client = k8s_client
        self._resource_optimizer = ResoureOptimizer(self._job_uuid)
        self._lock = threading.Lock()
        self.optimized_ps_mem = False
        self.optimize_worker_sampled = False
        self._job_stage = JobOptStage.CREATE

    def _init_job_resource_from_easydl(self):
        plan = self._resource_optimizer.get_optimization_plan(self._job_stage)
        if not plan:
            logger.info("Use the default plan to start the job")
            plan = get_default_plan()
        self._job_stage = JobOptStage.WORKER_INITIAL

        if (
            _dlrover_context.easydl_worker_enabled
        ) and NodeType.WORKER in plan.resource.task_group_resources:
            num, cpu, mem = extract_resource_from_plan(plan, NodeType.WORKER)
            num, cpu, mem = self._check_ignore_original_worker_resource(
                num, cpu, mem
            )
            self._worker_resource.update(num, cpu, mem)

        self._update_ps_source(plan)

    def _update_ps_source(self, plan):
        if NodeType.PS not in plan.resource.task_group_resources:
            return

        count, cpu, mem = extract_resource_from_plan(plan, NodeType.PS)
        if _dlrover_context.easydl_ps_enabled:
            count, cpu, mem = self._check_ignore_original_ps_resource(
                count, cpu, mem
            )
            self._ps_resource.update(count, cpu, mem)

    def optimize_worker_resource(self):
        plan = self._get_worker_resource_at_init_phase()
        if plan and NodeType.WORKER in plan.resource.task_group_resources:
            num, cpu, mem = extract_resource_from_plan(plan, NodeType.WORKER)
            self._worker_resource.update(num, cpu, mem)

    def get_worker_resource(self):
        return (
            self._worker_resource.num,
            self._worker_resource.cpu,
            self._worker_resource.mem,
        )

    def init_job_resource(self, typed_pod_config):
        """Adjust the initial resource of typed pods by EasyDL.
        Args:
            typed_pod_config: elasticdl.python.master.pod_info.TypedPodConfig
                it saves the pod configuration include pod number, memory
                and cpu.
        """
        self._init_job_resource_from_easydl()
        typed_pod_config.update_typed_pod_config(
            NodeType.WORKER,
            self._worker_resource.num,
            self._worker_resource.cpu,
            self._worker_resource.mem,
        )

        evaluator_config = typed_pod_config.get_typed_resource_config(
            NodeType.EVALUATOR
        )
        if evaluator_config.get_cpu() < NodeResourceLimit.MIN_VALID_CPU:
            evaluator_config.update_cpu(self._worker_resource.cpu)
        min_memory = NodeResourceLimit.MIN_VALID_MEMORY
        if evaluator_config.get_memory_mi() < min_memory:
            evaluator_config.update_memory(self._worker_resource.mem)

        typed_pod_config.update_typed_pod_config(
            NodeType.PS,
            self._ps_resource.num,
            self._ps_resource.cpu,
            self._ps_resource.mem,
        )
        return typed_pod_config

    def adjust_oom_worker_resource(self, pod_info):
        """Increment the memory to launch worker. The new memory
        is max(1.5 * memory, the memory set by users).

        Args:
            pod_info: PodInfo object
        """
        cur_mem = pod_info.resource_config.get_memory_mi()
        if (
            _dlrover_context.easydl_worker_enabled
            and self._job_stage == JobOptStage.WORKER_INITIAL
        ):
            plan = self._resource_optimizer.get_oom_resource_plan(
                [pod_info.name], JobOptStage.CREATE
            )
            if plan:
                _, _, memory = extract_resource_from_plan(
                    plan, NodeType.WORKER
                )
                self._worker_resource.mem = max(
                    self._worker_resource.mem, memory
                )
        else:
            self.optimize_worker_resource()
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        new_mem = int(
            max(
                self._worker_resource.mem,
                cur_mem,
                self._original_worker_resource.mem,
            )
        )
        pod_info.resource_config.update_memory(new_mem)
        logger.info(
            "Increment the memory of %s %s to %s",
            pod_info.type,
            pod_info.name,
            pod_info.resource_config.get_memory_mi(),
        )

    def adjust_oom_ps_resource(self, pod_info):
        plan = self._resource_optimizer.get_oom_resource_plan(
            [pod_info.name], JobOptStage.PS_INITIAL
        )
        if plan:
            count, _, memory = extract_resource_from_plan(plan, NodeType.PS)
            if (
                count > 0
                and self._ps_resource.num < NodeResourceLimit.MAX_PS_NUM
            ):
                self._verify_optimized_group_resource(plan, NodeType.PS)
                plan = adjust_plan_by_context(plan)
                return plan
            self._ps_resource.mem = max(self._ps_resource.mem, memory)
        cur_mem = pod_info.resource_config.get_memory_mi()
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        new_mem = int(
            max(
                self._ps_resource.mem, cur_mem, self._original_ps_resource.mem,
            )
        )
        pod_info.resource_config.update_memory(new_mem)
        logger.info(
            "Increment the memory of %s %s to %s",
            pod_info.type,
            pod_info.name,
            pod_info.resource_config.get_memory_mi(),
        )

    def get_job_resource_plan(self):
        plan = None
        if self._job_stage == JobOptStage.WORKER_INITIAL:
            plan = self._get_worker_resource_at_init_phase()
            self._job_stage = JobOptStage.PS_INITIAL
        elif self._job_stage == JobOptStage.PS_INITIAL:
            plan = self._get_ps_resource_plan()
            self._job_stage = JobOptStage.RUNNING
        elif self._job_stage == JobOptStage.RUNNING:
            plan = self._get_ps_resource_plan()
            if check_resource_plan_empty(plan):
                plan = self._get_worker_resource_at_running()
        if check_resource_plan_empty(plan):
            return None

        if NodeType.WORKER in plan.resource.task_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.WORKER)

        if plan and NodeType.PS in plan.resource.task_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.PS)

        plan = adjust_plan_by_context(plan)
        return plan

    def _get_worker_resource_at_running(self):
        if not self.optimize_worker_sampled:
            plan = self._get_worker_resource_at_sample_phase()
            self.optimize_worker_sampled = True
        else:
            plan = self._get_worker_resource_at_stable_phase()
        return plan

    def _get_worker_resource_at_init_phase(self):
        optimizer_config = {}
        optimizer_config[
            _WORKER_OPTIMIZE_PHASE
        ] = OptimizeWorkerPhase.INITIAL
        plan = self._resource_optimizer.get_optimization_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan:
            logger.info("No any plan to initialize the number of worker")
            return

        return plan

    def _get_worker_resource_at_sample_phase(self):
        optimizer_config = {}
        optimizer_config[
            _WORKER_OPTIMIZE_PHASE
        ] = OptimizeWorkerPhase.SAMPLE
        plan = self._resource_optimizer.get_optimization_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_worker_resource_at_stable_phase(self):
        optimizer_config = {}
        optimizer_config[
            _WORKER_OPTIMIZE_PHASE
        ] = OptimizeWorkerPhase.STABLE
        plan = self._resource_optimizer.get_optimization_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_ps_resource_plan(self):
        optimizer_config = {}
        plan = self._resource_optimizer.get_optimization_plan(
            self._job_stage, optimizer_config
        )
        return plan

    def _verify_optimized_group_resource(self, plan, pod_type):
        group_resource = plan.resource.task_group_resources[pod_type]
        num, cpu, mem = extract_resource_from_plan(plan, pod_type)
        if pod_type == NodeType.WORKER:
            num, cpu, mem = self._check_ignore_original_worker_resource(
                num, cpu, mem
            )
            self._worker_resource.num = num
            self._worker_resource.cpu = cpu
            self._worker_resource.mem = mem
        elif pod_type == NodeType.PS:
            num, cpu, mem = self._check_ignore_original_ps_resource(
                num, cpu, mem
            )
            self._ps_resource.num = min(num, NodeResourceLimit.MAX_PS_NUM)
            self._ps_resource.cpu = cpu
            self._ps_resource.mem = mem
        group_resource.count = num
        group_resource.resource.cpu = cpu
        group_resource.resource.memory = mem

    def _check_ignore_original_worker_resource(self, num, cpu, mem):
        """Abandon the optimization result if users have set the resource."""
        #  Users may worry about that the increasing number of worker hurts the
        #  accuracy, so the max number of worker is the configuration.
        if self._original_worker_resource.num > 0:
            num = self._original_worker_resource.num
        if (
            self._original_worker_resource.mem
            >= NodeResourceLimit.MIN_VALID_MEMORY
        ):
            mem = self._original_worker_resource.mem
        if (
            self._original_worker_resource.cpu
            >= NodeResourceLimit.MIN_VALID_CPU
        ):
            cpu = self._original_worker_resource.cpu
        return num, cpu, mem

    def _check_ignore_original_ps_resource(self, num, cpu, mem):
        """Abandon the optimization result if users have set the resource."""
        if self._original_ps_resource.num > 0:
            num = self._original_ps_resource.num
        if (
            self._original_ps_resource.mem
            >= NodeResourceLimit.MIN_VALID_MEMORY
        ):
            mem = self._original_ps_resource.mem
        if self._original_ps_resource.cpu >= NodeResourceLimit.MIN_VALID_CPU:
            cpu = self._original_ps_resource.cpu
        return num, cpu, mem

    def get_job_resource_quota(self):
        plan = self._resource_optimizer.get_estimated_resource_plan()
        quota = {}
        for task_type, group in plan.resource.task_group_resources.items():
            if group.count <= 0:
                continue
            stats = {
                "total_cpu": 0,
                "total_memory": 0,
                "total": 0,
                "active": 0,
                "succeeded": 0,
                "failed": 0,
            }
            stats["total"] = int(group.count)
            stats["total_cpu"] = int(group.resource.cpu * group.count)
            stats["total_memory"] = int(
                group.resource.memory * group.count / 1024 / 1024
            )
            stats["active"] = int(group.count)
            quota[task_type] = stats
        return quota
