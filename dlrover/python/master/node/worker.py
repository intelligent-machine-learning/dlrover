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
from typing import Dict, List

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.training_node import (
    ALIVE_STATUS,
    TrainingNodeManager,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.scaler.base_scaler import ScalePlan


class ChiefManager(TrainingNodeManager):
    def __init__(
        self,
        chief_nodes: Dict[int, Node],
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            chief_nodes: A dictionary where the key is the index of
                chief and the value is a Node instance.
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of a chief.
            new_service_fn: A callable function to generate a server name of
                chief.
            new_node_name_fn: A callable function to generate a node name of
                chief.
        """
        super(ChiefManager, self).__init__(chief_nodes, new_node_name_fn)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn

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
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            evaluator_nodes: A dictionary where the key is the index of
                evaluator and the value is a Node instance.
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of an evaluator.
            new_service_fn: A callable function to generate a server name of
                evaluator.
            new_node_name_fn: A callable function to generate a node name of
                evaluator.
        """
        super(EvaluatorManager, self).__init__(
            evaluator_nodes, new_node_name_fn
        )
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn

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
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            worker_nodes: A dictionary where the key is the index of worker
                and the value is a Node instance.
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of worker.
            new_service_fn: A callable function to generate a server name of
                worker.
            new_node_name_fn: A callable function to generate a node name of
                worker.
            use_ddp: bool, whether workers use DDP to train a model.
        """
        super(WorkerManager, self).__init__(worker_nodes, new_node_name_fn)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn

    def adjust_worker(self, worker_resource: NodeGroupResource):
        plan = ScalePlan()
        num = worker_resource.count
        logger.info(
            "Adjust worker resource to {}, {}, {}".format(
                num,
                worker_resource.node_resource.cpu,
                worker_resource.node_resource.memory,
            )
        )
        alive_workers = []
        completed_worker_num = 0
        for worker in self._nodes.values():
            if worker.status in ALIVE_STATUS:
                alive_workers.append(worker)
            elif worker.status in [NodeStatus.SUCCEEDED, NodeStatus.FINISHED]:
                completed_worker_num += 1
        alive_num = len(alive_workers)
        with self._lock:
            if num > alive_num + completed_worker_num:
                plan = self._scale_up_workers(
                    num - alive_num - completed_worker_num
                )
            elif num < alive_num:
                plan = self._scale_down_workers(alive_num - num, alive_workers)
        return plan

    def _scale_up_workers(self, up_num):
        """Launch up_num workers."""
        plan = ScalePlan()
        for _ in range(up_num):
            worker_id = next(self._node_id_iter)
            task_id = next(self._rank_id_iter)
            worker_resource = self._job_resource.get_node_group_resource(
                NodeType.WORKER
            ).node_resource
            service_addr = self._new_service_fn(NodeType.WORKER, task_id)
            new_node = Node(
                NodeType.WORKER,
                node_id=worker_id,
                rank_index=task_id,
                name=self._new_node_name_fn(NodeType.WORKER, worker_id),
                max_relaunch_count=self._max_relaunch_num,
                config_resource=copy.deepcopy(worker_resource),
                service_addr=service_addr,
            )
            self._nodes[worker_id] = new_node
            logger.info("Create worker %s", self._nodes[worker_id])
            plan.launch_nodes.append(new_node)
        return plan

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
                plan.remove_nodes.append(worker)
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
                    plan.remove_nodes.append(worker)
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
                plan.remove_nodes.append(worker)
        return plan

    def remove_noncritical_worker(self, worker_id):
        if self._nodes[worker_id].critical:
            logger.info("Skip the critical worker %s", worker_id)
        else:
            return self.remove_node(worker_id)

    def migrate_workers(self, workers: Dict[str, NodeResource]):
        """Migrate workers with the new resource"""
        plan = ScalePlan()
        for name, resource in workers.items():
            old_node_id = int(name.split("-")[-1])
            old_node = self._nodes[old_node_id]
            node_id = next(self._node_id_iter)
            task_id = old_node.rank_index
            new_node = Node(
                NodeType.WORKER,
                node_id,
                config_resource=resource,
                status=NodeStatus.INITIAL,
                rank_index=task_id,
                name=self._new_node_name_fn(NodeType.WORKER, node_id),
            )
            self._nodes[node_id] = new_node
            plan.launch_nodes.append(new_node)
            plan.remove_nodes.append(old_node)
        return plan

    def remove_not_joined_rdzv_workers(self, worker_ranks: List[int]):
        """Remove workers which do not participate in the training.
        Args:
            worker_ranks: The rank of worker which does not join rendezvous.
        """
        plan = ScalePlan()
        for node_id, node in self._nodes.items():
            if node.rank_index in worker_ranks:
                p = self.remove_node(node.id)
                self._nodes[node_id].relaunchable = False
                if p:
                    plan.merge(p)
        return plan

    def has_failed_worker(self):
        """Check whether there is failed worker except evicted workers."""
        for worker in self._nodes.values():
            if worker.exit_reason == NodeExitReason.FATAL_ERROR:
                return True
        return False

    def wait_worker_restart(self):
        """Check whether there are workers tha have remaining retries."""
        for worker in self._nodes.values():
            if (
                worker.exit_reason == NodeExitReason.KILLED
                and worker.relaunch_count < worker.max_relaunch_count
            ):
                return True
        return False
