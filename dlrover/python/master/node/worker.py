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
import itertools
from typing import Dict, List

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.training_node import (
    ALIVE_STATUS,
    TrainingNodeManager,
)
from dlrover.python.master.resource.job import JobResourceConfig
from dlrover.python.master.scaler.base_scaler import ScalePlan, LaunchNode


class ChiefManager(TrainingNodeManager):
    def __init__(
        self,
        chief_nodes: Dict[int, Node],
        job_resource: JobResourceConfig,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            k8s_client: The client to connect the k8s cluster.
            worker_pods: A dictionary where the key is the index of worker pod
                and the value is the PodInfo instance of PS pod.
            typed_pod_config: The ps/worker/evaluator configuration.
            max_relaunch_num: The maximum relaunch number of PS.
            command: The command of worker pods.
        """
        super(ChiefManager, self).__init__(chief_nodes)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn
        self._new_node_name_fn = new_node_name_fn

    def is_chief_running(self):
        """The chief worker with id=0 is responsible to initialize
        variables in TensorFlow 1.x PS strategy"""
        for node in self._nodes.values():
            if node.status == NodeStatus.RUNNING:
                return True
        return False


class EvaluatorManager(TrainingNodeManager):
    def __init__(
        self,
        evaluator_nodes: Dict[int, Node],
        job_resource: JobResourceConfig,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            k8s_client: The client to connect the k8s cluster.
            worker_pods: A dictionary where the key is the index of worker pod
                and the value is the PodInfo instance of PS pod.
            typed_pod_config: The ps/worker/evaluator configuration.
            max_relaunch_num: The maximum relaunch number of PS.
            command: The command of worker pods.
        """
        super(EvaluatorManager, self).__init__(evaluator_nodes)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn
        self._new_node_name_fn = new_node_name_fn

    def is_chief_running(self):
        """The chief worker with id=0 is responsible to initialize
        variables in TensorFlow 1.x PS strategy"""
        for node in self._nodes.values():
            if node.status == NodeStatus.RUNNING:
                return True
        return False


class WorkerManager(TrainingNodeManager):
    def __init__(
        self,
        worker_nodes: Dict[int, Node],
        job_resource: JobResourceConfig,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
        use_ddp=False,
    ):
        """
        Args:
            k8s_client: The client to connect the k8s cluster.
            worker_pods: A dictionary where the key is the index of worker pod
                and the value is the PodInfo instance of PS pod.
            typed_pod_config: The ps/worker/evaluator configuration.
            max_relaunch_num: The maximum relaunch number of PS.
            command: The command of worker pods.
        """
        super(WorkerManager, self).__init__(worker_nodes)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn
        self._new_node_name_fn = new_node_name_fn
        self._use_ddp = use_ddp
        worker = job_resource.get_node_group_resource(NodeType.WORKER)
        self._task_id_iter = itertools.count(worker.count)

    def adjust_worker(self, num, cpu, mem):
        logger.info(
            "Adjust worker resource to {}, {}, {}".format(num, cpu, mem)
        )
        alive_workers = []
        for worker in self._nodes.values():
            if worker.status in ALIVE_STATUS:
                alive_workers.append(worker)
        alive_num = len(alive_workers)
        with self._lock:
            if num > alive_num:
                return self._scale_up_workers(num - alive_num)
            elif num < alive_num:
                return self._scale_down_workers(alive_num - num, alive_workers)

    def _scale_up_workers(self, up_num):
        """Launch up_num workers."""
        for _ in range(up_num):
            worker_id = next(self._node_id_iter)
            task_id = next(self._task_id_iter)
            worker_resource = self._job_resource.get_node_group_resource(
                NodeType.WORKER
            )
            service_addr = self._new_service_fn(NodeType.WORKER, task_id)
            self._nodes[worker_id] = Node(
                NodeType.WORKER,
                node_id=worker_id,
                task_index=task_id,
                name=self._new_node_name_fn(NodeType.WORKER, worker_id),
                max_relaunch_count=self._max_relaunch_num,
                config_resource=copy.deepcopy(worker_resource),
                service_addr=service_addr,
            )

    def _scale_down_workers(self, down_num, running_workers: List[Node]):
        """Remove down_num running workers"""
        plan = ScalePlan()
        for worker in reversed(running_workers):
            if down_num <= 0:
                break
            if not worker.critical:
                worker.relaunchable = False
                worker.is_released = True
                down_num -= 1
            plan.remove_nodes.append(worker.name)
        return plan

    def delete_exited_workers(self):
        """Delete failed, succeed, finished workers."""
        plan = ScalePlan()
        with self._lock:
            for worker in self._nodes.values():
                if (
                    worker.status
                    in [
                        NodeStatus.FAILED,
                        NodeStatus.SUCCEEDED,
                        NodeStatus.FINISHED,
                    ]
                    and not worker.is_released
                ):
                    worker.is_released = True
                    plan.remove_nodes.append(worker.name)
        return plan

    def delete_running_workers(self):
        plan = ScalePlan()
        for worker in self._nodes.values():
            if not worker.critical and worker.status in [
                NodeStatus.RUNNING,
                NodeStatus.PENDING,
                NodeStatus.INITIAL,
            ]:
                worker.relaunchable = False
                logger.info(
                    "Remove the worker %s after the worker-0 completed",
                    worker.name,
                )
                worker.is_released = True
                plan.remove_nodes.append(worker.name)
        return plan

    def remove_noncritical_worker(self, worker_id):
        if self._nodes[worker_id].critical:
            logger.info("Skip the critical worker %s", worker_id)
        else:
            return self.remove_node(worker_id)
