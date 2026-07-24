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
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeResource, NodeGroupResource
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.node.worker import WorkerManager
from dlrover.python.master.resource.job import (
    JobResource,
    JobResourceOptimizer,
)
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler

_dlrover_context = Context.singleton_instance()


def new_job_resource_scaler(
    job_strategy,
    job_resource: JobResource,
    job_optimizer: JobResourceOptimizer,
    perf_monitor: PerfMonitor,
    worker_manager: WorkerManager,
    node_scaler: Scaler,
):
    if job_strategy == DistributionStrategy.ALLREDUCE:
        return AllreduceTrainingResourceScaler(
            job_resource,
            job_optimizer,
            perf_monitor,
            worker_manager,
            node_scaler,
        )
    else:
        raise ValueError("No job auto scaler for %s", job_strategy)


class JobResourceScaler(metaclass=ABCMeta):
    """JobResourceScaler scale up/down nodes resource of job."""

    def __init__(
        self,
        job_resource: JobResource,
        job_optimizer: JobResourceOptimizer,
        perf_monitor: PerfMonitor,
        node_scaler: Scaler,
        scale_interval: int,
    ):
        self._job_resource = job_resource
        self._job_optimizer = job_optimizer
        self._perf_monitor = perf_monitor
        self._scaler = node_scaler
        self._scale_interval = scale_interval
        self._job_context = get_job_context()

        self._suggested_stop = False
        self._autoscaling_started = False
    
        self._exec_opt_resource_ready = False
        self._save_ckpt_status = False

    def suggested_stop(self):
        return self._suggested_stop

    @abstractmethod
    def start_resource_scaling(self, scale_action: str):
        """Start vertical elastic nodes of a job"""
        pass

    @abstractmethod
    def execute_job_optimization_plan(self, plan: ResourcePlan):
        """Scale nodes of a job by a ResourcePlan"""
        if plan and not plan.empty():
            logger.info(f"Execute job optimization plan: {plan.to_json()}.")

    @abstractmethod
    def exec_opt_res_plan_ready(self):
        """Set the status of the job to start executing the optimization plan"""
        pass

    @abstractmethod
    def set_save_ckpt_status_by_vertical_elastic(self, save_ckpt_ready: bool):
        """set save_ckpt_status by vertical elastic"""
        pass

    @abstractmethod
    def set_param_tunning_ready(self, param_tunning_ready: bool):
        """set param tunning ready by vertical elastic"""
        pass

class AllreduceTrainingResourceScaler(JobResourceScaler):
    """Scale a Job resource with Allreduce strategy"""

    def __init__(
        self,
        job_resource: JobResource,
        job_optimizer: JobResourceOptimizer,
        perf_monitor: PerfMonitor,
        worker_manager: WorkerManager,
        node_scaler: Scaler,
    ) -> None:
        super().__init__(
            job_resource,
            job_optimizer,
            perf_monitor,
            node_scaler,
            1800,
        )
        self._worker_manager = worker_manager
        self.scale_action: str = ""
        self.last_plan_time = 0
        self.opt_num = 0

    def get_job_nodes(self, node_type=""):
        if node_type == "":
            return self._job_context.job_nodes()
        return self._job_context.job_nodes_by_type(node_type)

    def start_resource_scaling(self, scale_action: str):
        self.scale_action = scale_action
        logger.info(
            f"Start Resource scaling! scale_action is {self.scale_action}, " 
            f"enable_elastic_resource is {_dlrover_context.enable_elastic_resource}"
        )

        if not _dlrover_context.enable_elastic_resource:
            logger.warning(f"enable_elastic_resource is False, skip scaling.")
            return
        
        threading.Thread(
            target=self._adjust_running_worker_resource,
            name="allreduce-worker-resource-scaler",
            daemon=True,
        ).start()


    def _adjust_running_worker_resource(self):
        """Adjust the resource of worker."""

        opt_interval = _dlrover_context.vertical_elastic_opt_interval
        max_opt_num = _dlrover_context.vertical_elastic_max_opt_num
        ckpt_wait_time = _dlrover_context.ckpt_save_status_check_max_wait_time
        ckpt_check_interval = _dlrover_context.ckpt_save_status_check_interval
        log_step = _dlrover_context.log_interval_check_nums
        logger.info(f"vertical elastic control params:\n"
            f"  vertical elastic opt_interval         is: {opt_interval}\n"
            f"  vertical elastic max_opt_num          is: {max_opt_num}\n"
            f"  ckpt_save_status_check_max_wait_time  is: {ckpt_wait_time}\n"
            f"  ckpt_save_status_check_interval       is: {ckpt_check_interval}\n"
            f"  log_interval_check_nums               is: {log_step}\n"
        )

        try:
            # Control the interval to query plans
            current_time = time.time()
            elapsed = current_time - self.last_plan_time
        
            logger.info(f"Elapsed: {elapsed:.1f}s,"
                f"current_time: {current_time}s,"
                f"last_plan_time: {self.last_plan_time}s,"
                f"opt_interval Config: {opt_interval}s,"
                f"Diff: {opt_interval - elapsed:.1f}s"
            )
            
            if self.opt_num > max_opt_num:
                logger.warning(f"Reached max optimization attempts ({max_opt_num}). Skip scaling.")
                return

            manual_scale_switch = _dlrover_context.configmap_manual_scale_switch
            if elapsed <= opt_interval:
                logger.warning(f"The time elapsed since the last execution is too short; Manual scale switch is {manual_scale_switch}.")
                if manual_scale_switch == "on":
                    logger.warning(f"Continue manual scaling.")
                else:
                    logger.warning(f"Skip scaling.")
                    return
            
            if not self._try_execute_optimization(ckpt_wait_time, ckpt_check_interval, log_step):
                return
            
            self.opt_num += 1
            self.last_plan_time = time.time()

            logger.info(f"Optimization round {self.opt_num} completed successfully.")
        except Exception as e:
            logger.error(
                f"Failed to worker resource scale for AllReduce Training: {e}"
            )
            self._exec_opt_resource_ready = False
    
    def _try_execute_optimization(self, ckpt_wait_time, ckpt_check_sec, log_step):
        """algorithm：get plan -> wait CKPT -> exec plan。"""
        # A. Get plan
        alive_num = self._get_alive_worker_num()
        self._job_optimizer.set_alive_node_num(alive_num)
        
        plan = self._job_optimizer.get_job_resource_plan(self.scale_action)
        if not plan or plan.empty():
            logger.warning("No valid resource plan generated.")
            return False

        if not getattr(_dlrover_context, 'enable_elastic_resource', False):
            return False

        # B. Wait Checkpoint finish
        self._exec_opt_resource_ready = True
        if not self._wait_for_ckpt_ready(ckpt_wait_time, ckpt_check_sec, log_step):
            self._exec_opt_resource_ready = False
            return False

        # C. Exec plan
        try:
            self.execute_job_optimization_plan(plan)
            # self._exec_opt_resource_ready = False
            return True
        except Exception as e:
            logger.error(f"Failed to execute plan: {e}", exc_info=True)
            self._exec_opt_resource_ready = False
            return False

    def _wait_for_ckpt_ready(self, max_wait, check_interval, log_step):
        """Wait ckpt ready"""

        start_time = time.time()
        count = 0
        
        while not self._save_ckpt_status:
            if not _dlrover_context.enable_elastic_resource:
                logger.warning(f"Enable elastic resource is {_dlrover_context.enable_elastic_resource}, Resource Scaling stopped while waiting for CKPT.")
                return False
                
            if (time.time() - start_time) > max_wait:
                logger.warning("Waiting for CKPT save timed out.")
                return False
            
            count += 1
            if count % log_step == 0:
                logger.info(f"Waiting for CKPT... ({time.time() - start_time:.1f}s)")
            
            time.sleep(check_interval)
        
        logger.info("CKPT save completed. Proceeding with optimization.")
        return True

    def _get_alive_worker_num(self):
        worker_num = 0
        workers = self.get_job_nodes(NodeType.WORKER)
        for _, worker in workers.items():
            if worker.status in [
                NodeStatus.RUNNING,
                NodeStatus.PENDING,
                NodeStatus.INITIAL,
                NodeStatus.SUCCEEDED,
            ]:
                worker_num += 1
        return worker_num

    def execute_job_optimization_plan(self, plan: ResourcePlan):
        """Execute the optimization plan of the training job.
        The plan may adjust the number of PS and workers or
        adjust the cpu and memory of nodes.
        """
        super().execute_job_optimization_plan(plan)
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
                    group.node_resource.gpu_type,
                    group.node_resource.gpu_num,
                )
                group = self._job_resource.get_node_group_resource(node_type)
                self._perf_monitor.set_target_worker_num(group.count)
                if _dlrover_context.enable_elastic_resource:
                    worker_plan = self.get_worker_plan_with_vertical_elastic(group)
                else:
                    worker_plan = self.get_worker_plan_with_horizontal_elastic(group)
                scale_plan.merge(worker_plan)
        self._scaler.scale(scale_plan)
        return scale_plan
    
    def exec_opt_res_plan_ready(self):
        return self._exec_opt_resource_ready
    
    def set_save_ckpt_status_by_vertical_elastic(self, save_ckpt_ready: bool):
        self._save_ckpt_status = save_ckpt_ready

    def set_param_tunning_ready(self, param_tunning_ready: bool):
        if param_tunning_ready:
            logger.info("Param tunning is completed, set exec_opt_resource_ready to False")
            # self._exec_opt_resource_ready = False

    def get_worker_plan_with_vertical_elastic(self, worker_resource: NodeGroupResource):
        logger.info("[AllReduce] start to adjust worker resource by vertical elastic")

        worker_plan = ScalePlan()
        
        sacle_down_worker_plan = self._worker_manager.adjust_worker_by_scale_operate(worker_resource, "scale_down")
        worker_plan = self._worker_manager.adjust_worker_by_scale_operate(worker_resource, "scale_up")
        
        if sacle_down_worker_plan:
            worker_plan.remove_nodes.extend(sacle_down_worker_plan.remove_nodes)
        logger.info("[AllReduce] after adjust worker, worker plan is: %s", worker_plan.to_json())
        return worker_plan

    def get_worker_plan_with_horizontal_elastic(self, worker_resource: NodeGroupResource):
        return self._worker_manager.adjust_worker(worker_resource)