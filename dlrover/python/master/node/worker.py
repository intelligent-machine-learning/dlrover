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
import time
from typing import Dict, List, Tuple

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.training_node import (
    ALIVE_STATUS,
    TrainingNodeManager,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.scaler.base_scaler import ScalePlan

_dlrover_context = Context.singleton_instance()


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
        """
        super(WorkerManager, self).__init__(worker_nodes, new_node_name_fn)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn
        # the required nodes number, format: (min_required, max_required)
        self._nodes_required = (0, 0)
        self._last_insufficient_nodes_timestamp = 0

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
        for worker in self._nodes.values():
            if worker.status in ALIVE_STATUS:
                alive_workers.append(worker)
        alive_num = len(alive_workers)
        with self._lock:
            if num > alive_num:
                plan = self._scale_up_workers(num - alive_num)
            elif num < alive_num:
                plan = self._scale_down_workers(alive_num - num, alive_workers)
        return plan

    def _scale_up_workers(self, up_num):
        """Launch up_num workers."""
        plan = ScalePlan()
        for _ in range(up_num):
            worker_id = next(self._node_id_iter)
            task_id = next(self._node_rank_iter)
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
                    "Remove the worker %s after the chief completed",
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
            if old_node.critical:
                continue
            old_node.migrated = True
            old_node.relaunchable = False
            old_node.is_released = True
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

    def has_exited_worker(self):
        """Check whether there is exited worker except evicted workers."""
        for worker in self._nodes.values():
            if (
                worker.exit_reason == NodeExitReason.FATAL_ERROR
                or worker.status == NodeStatus.SUCCEEDED
            ):
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

    def verify_restarting_training(self, node_id):
        """
        Verify if the worker requires restarting the training process.
        The worker will restart the training processes if any of the
        following conditions are met:
            1. RestartTrain action in the Pod annotations.
            2. One training process crashes in the worker.

        args:
            node_id: the worker node id.

        Return:
            bool
        """
        restart = False
        worker = self._nodes[node_id]
        if not worker.is_released:
            restart = worker.restart_training
            # Set False to avoid restart repeatedly.
            worker.restart_training = False
        return restart

    def is_training_hang_by_pending(self) -> bool:
        """
        To prevent training hanging by pending nodes. Should exit when there is
        inextricable pending issue.

        There is 3 main conditions:
        1: exist pending nodes
        2: alive nodes number consistently lower than the min nodes requires
        3: 1+2 last for a certain time

        Return:
            bool
        """

        if not self.has_node_required_info():
            return False

        # pending time as timeout for now
        timeout = _dlrover_context.seconds_to_wait_pending_pod
        cur_nodes = list(self._nodes.values())

        # collect pending and running nodes
        pending_nodes: List[Node] = []
        running_nodes: List[Node] = []
        for node in cur_nodes:
            if node is None or node.is_released or node.create_time is None:
                continue

            if node.status in [NodeStatus.PENDING, NodeStatus.INITIAL]:
                pending_nodes.append(node)
            elif node.status == NodeStatus.RUNNING:
                running_nodes.append(node)

        # with condition 1 + 2
        if (
            len(pending_nodes) == 0
            or len(running_nodes) >= self.get_min_nodes_required()
        ):
            return False

        # with condition 3
        now = time.time()
        first_pending_node = min(
            pending_nodes, key=lambda x: x.create_time  # type: ignore
        )
        if not first_pending_node or not first_pending_node.create_time:
            return False

        if now - first_pending_node.create_time.timestamp() > timeout:
            logger.warning(
                f"Node {first_pending_node.name} "
                f"exceeded pending timeout: {timeout}s."
            )
            return True

        return False

    def is_training_hang_by_insufficient_worker(self) -> bool:
        """
        There is a small probability that due to unknown reason on the
        training side, some workers will exit normally at the end of training,
        while others will fail and restart. This function is to make sure EDL
        can hold such circumstance, end the job directly.

        Return:
            bool
        """

        if not self.has_node_required_info():
            return False

        # use twice pending time as timeout
        timeout = _dlrover_context.seconds_to_wait_pending_pod * 2
        cur_nodes = list(self._nodes.values())

        # collect available nodes
        available_nodes: List[Node] = []
        for node in cur_nodes:
            if not node.is_released and node.status in [
                NodeStatus.RUNNING,
                NodeStatus.PENDING,
                NodeStatus.INITIAL,
            ]:
                available_nodes.append(node)

        now = time.time()
        if len(available_nodes) < self.get_min_nodes_required():
            if self._last_insufficient_nodes_timestamp == 0:
                self._last_insufficient_nodes_timestamp = int(now)
                logger.warning(f"Job with insufficient nodes: {cur_nodes}.")
            else:
                if now - self._last_insufficient_nodes_timestamp > timeout:
                    logger.warning(
                        f"Job with insufficient nodes: {cur_nodes} "
                        f"lasts for more than {timeout}s."
                    )
                    return True
        else:
            self._last_insufficient_nodes_timestamp = 0

        return False

    def has_node_required_info(self):
        if self._nodes_required[0] and self._nodes_required[1]:
            return True
        return False

    def get_min_nodes_required(self) -> int:
        """Notice: it is meaningless when the result is 0."""

        return self._nodes_required[0]

    def get_max_nodes_required(self) -> int:
        """Notice: it is meaningless when the result is 0."""

        return self._nodes_required[1]

    def update_node_required_info(self, nodes_required: Tuple[int, int]):
        self._nodes_required = nodes_required
