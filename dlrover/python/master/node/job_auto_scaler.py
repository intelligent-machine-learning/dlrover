# Copyright 2023 The DLRover Authors. All rights reserved.
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
import time
from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.node.worker import WorkerManager
from dlrover.python.master.resource.job import (
    JobResource,
    JobResourceOptimizer,
)
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler

_dlrover_context = Context.singleton_instance()


def new_job_auto_scaler(
    job_strategy,
    job_resource: JobResource,
    job_nodes: Dict[str, Dict[int, Node]],
    job_optimizer: JobResourceOptimizer,
    speed_monitor: SpeedMonitor,
    ps_manager: ParameterServerManager,
    worker_manager: WorkerManager,
    node_scaler: Scaler,
):
    if job_strategy == DistributionStrategy.PS:
        return PSTrainingAutoScaler(
            job_resource,
            job_nodes,
            job_optimizer,
            speed_monitor,
            ps_manager,
            worker_manager,
            node_scaler,
        )
    elif job_strategy == DistributionStrategy.ALLREDUCE:
        return AllreduceTrainingAutoScaler(
            job_resource,
            job_nodes,
            job_optimizer,
            speed_monitor,
            worker_manager,
            node_scaler,
        )
    else:
        raise ValueError("No job auto scaler for %s", job_strategy)


class JobAutoScaler(metaclass=ABCMeta):
    """JobAutoScaler automatically scale up/down nodes of job."""

    @abstractmethod
    def start_auto_scaling(self):
        """Start auto-scaling nodes of a job"""
        pass

    @abstractmethod
    def stop_auto_scaling(self):
        """Stop auto-scaling nodes of a job"""
        pass

    @abstractmethod
    def execute_job_optimization_plan(self, plan: ResourcePlan):
        """Scale nodes of a job by a ResourcePlan"""
        pass


class PSTrainingAutoScaler(JobAutoScaler):
    """AutoScale a Job using Async-SGD with ParamterServer strategy"""

    def __init__(
        self,
        job_resource: JobResource,
        job_nodes: Dict[str, Dict[int, Node]],
        job_optimizer: JobResourceOptimizer,
        speed_monitor: SpeedMonitor,
        ps_manager: ParameterServerManager,
        worker_manager: WorkerManager,
        node_scaler: Scaler,
    ) -> None:
        self._job_resource = job_resource
        self._job_optimizer = job_optimizer
        self._speed_monitor = speed_monitor
        self._ps_manager = ps_manager
        self._worker_manager = worker_manager
        self._stop_autoscaling = False
        self._scaler = node_scaler
        self._job_nodes = job_nodes
        self._autoscaling_started = False

    def start_auto_scaling(self):
        """Start to auto-scale nodes to improve the training throughput."""
        if not self._autoscaling_started:
            logger.info("AutoScaling started!")
            self._autoscaling_started = True
            if (
                not _dlrover_context.auto_ps_enabled
                and not _dlrover_context.auto_worker_enabled
            ):
                return
            if self._speed_monitor:
                self._speed_monitor.set_target_worker_num(
                    self._job_resource.worker_num
                    + self._job_resource.chief_num
                )
                threading.Thread(
                    target=self._periodic_optimize_running_resource,
                    name="ps-autoscaler",
                    daemon=True,
                ).start()

    def stop_auto_scaling(self):
        self._stop_autoscaling = True

    def _periodic_optimize_running_resource(self):
        """Adjust job resource periodically and stop adjustment
        if there is a failed worker with the fatal error.
        """
        logger.info("Start to auto scale ")
        last_plan_time = 0
        opt_interval = _dlrover_context.seconds_interval_to_optimize
        while True:
            if self._stop_autoscaling:
                logger.info("Stop auto-scaling PS Trainign.")
                break
            if (
                self._speed_monitor.worker_adjustment_finished()
                # Control the interval to query plans
                and time.time() - last_plan_time > opt_interval
                and not self._ps_manager.exist_migrated_ps_nodes()
            ):
                plan = self._job_optimizer.get_job_resource_plan()
                if plan:
                    last_plan_time = time.time()
                self.execute_job_optimization_plan(plan)
            self._cut_timeout_pending_node_cpu()
            time.sleep(30)

    def execute_job_optimization_plan(self, plan: ResourcePlan):
        """Execute the optimization plan of the training job.
        The plan may adjust the number of PS and workers or
        adjust the cpu and memory of nodes.
        """
        scale_plan = ScalePlan()
        if not plan or plan.empty():
            return scale_plan
        for node_type, group in plan.node_group_resources.items():
            if group.count > 0:
                self._job_resource.update_node_group_resource(
                    node_type,
                    group.count,
                    group.node_resource.cpu,
                    group.node_resource.memory,
                )
                group = self._job_resource.get_node_group_resource(node_type)
                if node_type == NodeType.PS:
                    ps_plan = self._ps_manager.adjust_ps(group)
                    scale_plan.merge(ps_plan)
                    self._speed_monitor.reset_running_speed_monitor()
                elif node_type == NodeType.WORKER:
                    chief_num = len(self._job_nodes.get(NodeType.CHIEF, []))
                    worker_num = chief_num + group.count
                    self._speed_monitor.set_target_worker_num(worker_num)
                    worker_plan = self._worker_manager.adjust_worker(group)
                    scale_plan.merge(worker_plan)
        if len(plan.node_resources) > 0:
            migration_plan = self._migrate_nodes(plan.node_resources)
            scale_plan.merge(migration_plan)
        ps_addrs = self._ps_manager.get_ps_addrs()
        scale_plan.ps_addrs.extend(ps_addrs)
        self._scaler.scale(scale_plan)
        return scale_plan

    def _migrate_nodes(self, node_resources: Dict[str, NodeResource]):
        workers: Dict[str, NodeResource] = {}
        ps: Dict[str, NodeResource] = {}
        for name, resource in node_resources.items():
            type = name.split("-")[-2]
            if type == NodeType.WORKER:
                workers[name] = resource
            elif type == NodeType.PS:
                ps[name] = resource

        scale_plan = ScalePlan()
        if len(ps) > 0:
            plan = self._ps_manager.migrate_parameter_servers(ps)
            scale_plan.merge(plan)
            self._speed_monitor.reset_running_speed_monitor()

        if len(workers) > 0:
            plan = self._worker_manager.migrate_workers(workers)
            scale_plan.merge(plan)
        logger.info("Migration plan = %s", scale_plan.toJSON())
        return scale_plan

    def _cut_timeout_pending_node_cpu(self):
        """Cut down CPU cores of pending pod at the job starts"""
        if self._autoscaling_started:
            return
        if _dlrover_context.auto_ps_enabled:
            self._ps_manager.cut_pending_node_cpu()
        if _dlrover_context.auto_worker_enabled:
            self._worker_manager.cut_pending_node_cpu()


class AllreduceTrainingAutoScaler(JobAutoScaler):
    """AutoScale a Job using Async-SGD with ParamterServer strategy"""

    def __init__(
        self,
        job_resource: JobResource,
        job_nodes: Dict[str, Dict[int, Node]],
        job_optimizer: JobResourceOptimizer,
        speed_monitor: SpeedMonitor,
        worker_manager: WorkerManager,
        node_scaler: Scaler,
    ) -> None:
        self._job_resource = job_resource
        self._job_optimizer = job_optimizer
        self._speed_monitor = speed_monitor
        self._worker_manager = worker_manager
        self._stop_autoscaling = False
        self._scaler = node_scaler
        self._workers = job_nodes[NodeType.WORKER]
        self._autoscaling_started = False
        self._scale_interval = 1800

    def start_auto_scaling(self):
        """Start auto-scaling nodes of a job"""
        if not self._autoscaling_started:
            self._autoscaling_started = True
            if _dlrover_context.auto_worker_enabled:
                threading.Thread(
                    target=self._periodic_adjust_worker,
                    name="allreduce-autoscaler",
                    daemon=True,
                ).start()

    def _periodic_adjust_worker(self):
        """Periodicaly adjust the number of worker."""
        logger.info("Start to auto scale the number of workers.")
        while True:
            time.sleep(self._scale_interval)
            alive_num = self._get_alive_worker_num()
            self._job_optimizer.set_alive_node_num(alive_num)
            plan = self._job_optimizer.get_job_resource_plan()
            new_worker_num = plan.node_group_resources[NodeType.WORKER].count
            if new_worker_num <= alive_num:
                continue
            self.execute_job_optimization_plan(plan)

    def _get_alive_worker_num(self):
        worker_num = 0
        for _, worker in self._workers.items():
            if worker.status in [
                NodeStatus.RUNNING,
                NodeStatus.PENDING,
                NodeStatus.INITIAL,
                NodeStatus.SUCCEEDED,
            ]:
                worker_num += 1
        return worker_num

    def stop_auto_scaling(self):
        """Stop auto-scaling nodes of a job"""
        pass

    def execute_job_optimization_plan(self, plan: ResourcePlan):
        """Execute the optimization plan of the training job.
        The plan may adjust the number of PS and workers or
        adjust the cpu and memory of nodes.
        """
        scale_plan = ScalePlan()
        if not plan or plan.empty():
            return scale_plan
        for node_type, group in plan.node_group_resources.items():
            if node_type != NodeType.WORKER:
                continue
            if group.count > 0:
                self._job_resource.update_node_group_resource(
                    node_type,
                    group.count,
                    group.node_resource.cpu,
                    group.node_resource.memory,
                )
                group = self._job_resource.get_node_group_resource(node_type)
                self._speed_monitor.set_target_worker_num(group.count)
                worker_plan = self._worker_manager.adjust_worker(group)
                scale_plan.merge(worker_plan)
        self._scaler.scale(scale_plan)
        return scale_plan
