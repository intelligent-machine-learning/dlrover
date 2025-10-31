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
from typing import Dict, List, Tuple, Optional

from dlrover.python.common.constants import (
    DistributionStrategy,
    JobConstant,
    NodeEventType,
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
    is_all_nodes_pending_judgement,
    is_key_nodes_pending_judgement,
    skip_pending_judgement,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.scaler.base_scaler import ScalePlan

_dlrover_context = Context.singleton_instance()


class ChiefManager(TrainingNodeManager):
    def __init__(
        self,
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of a chief.
            new_service_fn: A callable function to generate a server name of
                chief.
            new_node_name_fn: A callable function to generate a node name of
                chief.
        """
        super(ChiefManager, self).__init__(NodeType.CHIEF, new_node_name_fn)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn

    def is_chief_running(self):
        """The chief worker with id=0 is responsible to initialize
        variables in TensorFlow 1.x PS strategy"""
        nodes = self._get_nodes()
        for node in nodes.values():
            if node.status == NodeStatus.RUNNING:
                return True
        return False


class EvaluatorManager(TrainingNodeManager):
    def __init__(
        self,
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of an evaluator.
            new_service_fn: A callable function to generate a server name of
                evaluator.
            new_node_name_fn: A callable function to generate a node name of
                evaluator.
        """
        super(EvaluatorManager, self).__init__(
            NodeType.EVALUATOR, new_node_name_fn
        )
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn

    def is_chief_running(self):
        """The chief worker with id=0 is responsible to initialize
        variables in TensorFlow 1.x PS strategy"""
        nodes = self._get_nodes()
        for node in nodes.values():
            if node.status == NodeStatus.RUNNING:
                return True
        return False


class WorkerManager(TrainingNodeManager):
    def __init__(
        self,
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of worker.
            new_service_fn: A callable function to generate a server name of
                worker.
            new_node_name_fn: A callable function to generate a node name of
                worker.
        """
        super(WorkerManager, self).__init__(NodeType.WORKER, new_node_name_fn)
        self._job_resource = job_resource
        self._max_relaunch_num = max_relaunch_num
        self._new_service_fn = new_service_fn
        # the required nodes number, format: (min_required, max_required)
        self._nodes_required = (0, 0, 0)
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
        nodes = self._get_mutable_nodes()
        for worker in nodes.values():
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
            worker_id = self.get_next_node_id()
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
            self._update_node(new_node)
            logger.info("Create worker %s", new_node)
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
        nodes = self._get_mutable_nodes()
        with self._lock:
            for worker in nodes.values():
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
        nodes = self._get_mutable_nodes()
        for worker in nodes.values():
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
        node = self._job_context.job_node(self._node_type, worker_id)
        if node is None:
            logger.error(f"not found node[{self._node_type}][{worker_id}]")
            return
        if node.critical:
            logger.info("Skip the critical worker %s", worker_id)
        else:
            return self.remove_node(worker_id)

    def migrate_workers(self, workers: Dict[str, NodeResource]):
        """Migrate workers with the new resource"""
        plan = ScalePlan()
        nodes = self._get_mutable_nodes()
        for name, resource in workers.items():
            old_node_id = int(name.split("-")[-1])
            old_node = nodes[old_node_id]
            if old_node.critical:
                continue
            old_node.migrated = True
            old_node.relaunchable = False
            old_node.is_released = True
            node_id = self.get_next_node_id()
            task_id = old_node.rank_index
            new_node = Node(
                NodeType.WORKER,
                node_id,
                config_resource=resource,
                status=NodeStatus.INITIAL,
                rank_index=task_id,
                name=self._new_node_name_fn(NodeType.WORKER, node_id),
            )
            self._update_node(new_node)
            plan.launch_nodes.append(new_node)
            plan.remove_nodes.append(old_node)
        return plan

    def remove_not_joined_rdzv_workers(self, worker_ranks: List[int]):
        """Remove workers which do not participate in the training.
        Args:
            worker_ranks: The rank of worker which does not join rendezvous.
        """
        plan = ScalePlan()
        nodes = self._get_mutable_nodes()
        for node_id, node in nodes.items():
            if node.rank_index in worker_ranks:
                p = self.remove_node(node.id)
                nodes[node_id].relaunchable = False
                if p:
                    plan.merge(p)
        return plan

    def has_exited_worker(self):
        """Check whether there is exited worker except evicted workers.

        If the worker has reported SUCCEEDED_EXITED, but been deleted
        by dlrover finally, the status will be DELETED instead of SUCCEEDED
        In such cases the worker should also be regard as exited worker
        """
        nodes = self._get_nodes()
        for worker in nodes.values():
            if (
                worker.exit_reason == NodeExitReason.FATAL_ERROR
                or worker.status == NodeStatus.SUCCEEDED
                or (
                    worker.status == NodeStatus.DELETED
                    and worker.get_reported_status()
                    == NodeEventType.SUCCEEDED_EXITED
                )
            ):
                logger.debug(
                    f"Worker {worker} has exited: {worker.exit_reason} {worker.status}"
                )
                return True
        return False

    def wait_worker_restart(self):
        """Check whether there are workers tha have remaining retries."""
        nodes = self._get_nodes()
        for worker in nodes.values():
            if (
                worker.exit_reason == NodeExitReason.KILLED
                and worker.relaunch_count < worker.max_relaunch_count
            ):
                logger.debug(f"Worker {worker} is restarting")
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
        worker = self._job_context.job_node(self._node_type, node_id)
        if worker is None:
            logger.error(f"not found worker-{node_id}")
            return False
        if not worker.is_released:
            restart = worker.restart_training
            # Set False to avoid restart repeatedly.
            worker.restart_training = False
            self._update_node(worker)
        return restart

    def find_pending_node_caused_training_hang(
        self, total_node_num, job_type
    ) -> Optional[str]:
        """
        To prevent training hang by pending workers. Should exit when there
        is inextricable pending issue.

        The fail strategy:
        torch:
        0: skip judgement
        1 and 2: all workers should be ready within the timeout period
        tf:
        0: skip judgement
        1: at least 1 worker(chief) should be ready within the timeout period
        2: all role nodes should be ready within the timeout period

        There is 3 main conditions for the judgement:
        1: exist pending nodes
        2: alive nodes number consistently lower than the min nodes requires
        3: 1+2 last for a certain time

        Args:
            total_node_num(int): Total node number master managed.
            job_type(str): Job type. Support AllReduceStrategy and
                ParameterServerStrategy for now.

        Return:
            str: return worker name if has pending
        """

        # fail strategy
        strategy = _dlrover_context.pending_fail_strategy

        # pending time as timeout for now
        timeout = self._get_pending_timeout()
        logger.debug(
            "Is training hang by pending with total worker "
            f"num: {total_node_num}, timeout: {timeout}, strategy: {strategy}."
        )
        if timeout <= 0 or skip_pending_judgement(strategy):
            return None

        # collect pending and running nodes
        cur_nodes = list(self._get_nodes().values())
        pending_workers: List[Node] = []
        running_workers: List[Node] = []
        for node in cur_nodes:
            if node is None or node.is_released or node.create_time is None:
                continue
            if node.status in [NodeStatus.PENDING, NodeStatus.INITIAL]:
                pending_workers.append(node)
            elif node.status == NodeStatus.RUNNING:
                running_workers.append(node)

        if not self.has_node_required_info() and total_node_num != len(
            pending_workers
        ):
            logger.debug(
                "Skip for no required nodes info and not all nodes pending."
            )
            return None

        if job_type == DistributionStrategy.ALLREDUCE or (
            job_type == DistributionStrategy.PS
            and is_all_nodes_pending_judgement(strategy)
        ):
            if 0 < len(pending_workers) == total_node_num:
                # all nodes pending
                logger.info(f"All workers pending: {pending_workers}.")
            else:
                # partial nodes pending
                # with condition 1 + 2
                if (
                    len(pending_workers) == 0
                    or len(running_workers) >= self.get_min_nodes_required()
                ):
                    logger.debug(
                        f"Skip for no pending workers: {pending_workers} "
                        f"or running workers: {running_workers} is greater "
                        f"than the min workers "
                        f"required: {self.get_min_nodes_required()}."
                    )
                    return None

            # with condition 3
            now = time.time()
            first_pending_worker = min(
                pending_workers,
                key=lambda x: x.create_time,  # type: ignore
            )

            if (
                first_pending_worker.create_time
                and now - first_pending_worker.create_time.timestamp()
                > timeout
            ):
                logger.warning(
                    f"Node {first_pending_worker.name} "
                    f"exceeded pending timeout: {timeout}s, "
                    f"job-type: {job_type}, strategy: {strategy}, "
                    f"running workers(size:{len(running_workers)})"
                    f": {running_workers}, "
                    f"pending workers(size:{len(pending_workers)})"
                    f": {pending_workers}, "
                    "min required nodes size"
                    f": {self.get_min_nodes_required()}."
                )
                return first_pending_worker.name
        elif (
            job_type == DistributionStrategy.PS
            and is_key_nodes_pending_judgement(strategy)
        ):
            # get worker 0
            pending_worker_0 = None
            for pending_worker in pending_workers:
                if pending_worker.rank_index == 0:
                    pending_worker_0 = pending_worker
                    break
            if not pending_worker_0 or not pending_worker_0.create_time:
                logger.debug(
                    "Skip for no pending worker0 or pending worker's "
                    f"create time is None: {pending_worker_0}."
                )
                return None

            now = time.time()
            if now - pending_worker_0.create_time.timestamp() > timeout:
                logger.warning(
                    f"Node {pending_worker_0.name} "
                    f"exceeded pending timeout: {timeout}s, "
                    f"job-type: {job_type}, strategy: {strategy}, "
                    f"running workers(size:{len(running_workers)})"
                    f": {running_workers}, "
                    f"pending workers(size:{len(pending_workers)})"
                    f": {pending_workers}."
                )
                return pending_worker_0.name
        return None

    def get_pending_node_groups(self, job_type):
        """
        Check if pod pending happened in a node group, only for torch
        training fail-over.
        If it happened, we should relaunch the node group

        Args:
            job_type: job type

        Returns:
            list of node group index

        """
        strategy = _dlrover_context.pending_fail_strategy
        timeout = self._get_group_pending_timeout()
        logger.debug(
            f"Is training hang by group pending: {timeout} {strategy} {job_type}."
        )

        pending_workers: List[Node] = []
        pending_groups: List[int] = []
        if (
            timeout <= 0
            or skip_pending_judgement(strategy)
            or job_type != DistributionStrategy.ALLREDUCE
        ):
            return []

        for group_key in self._job_context.job_node_groups_keys():
            node_group = self._job_context.job_node_group(group_key)
            for node in node_group.values():
                if (
                    node is None
                    or node.is_released
                    or node.create_time is None
                    or node.group_size is None
                    or node.group is None
                ):
                    logger.debug(f"Skip invalid node {node}")
                    continue

                logger.debug(f"Recheck node {node} for node group pending...")

                # ignore first startup, only check after node relaunch
                if node.relaunch_count > 0 and node.status in [
                    NodeStatus.PENDING,
                    NodeStatus.INITIAL,
                ]:
                    logger.info(
                        f"Detect pending node {node.id} with "
                        f"rank {node.rank_index} and group {node.group}: "
                        f"relaunch {node.relaunch_count}, status {node.status}"
                    )
                    pending_workers.append(node)

            if pending_workers:
                first_pending_worker = min(
                    pending_workers,
                    key=lambda x: x.create_time,  # type: ignore
                )
            else:
                first_pending_worker = None

            if (
                not first_pending_worker
                or not first_pending_worker.create_time
            ):
                logger.debug(f"Skip pending worker {first_pending_worker}")
                continue

            logger.info(
                f"Check first pending node {first_pending_worker.id} with "
                f"rank {first_pending_worker.rank_index} "
                f"and group {first_pending_worker.group}"
            )
            now = time.time()
            if now - first_pending_worker.create_time.timestamp() > timeout:
                logger.warning(
                    f"Node {first_pending_worker.name} with "
                    f"id {first_pending_worker.id} "
                    f"rank {first_pending_worker.rank_index} "
                    f"exceeded group pending timeout: {timeout}s, "
                    f"job-type: {job_type}, strategy: {strategy}, "
                    f"group: {first_pending_worker.group}, "
                    f"group_size: {first_pending_worker.group_size}, "
                    f"group_id: {first_pending_worker.group_id}."
                )
                pending_groups.append(first_pending_worker.group)

        return pending_groups

    def _get_insufficient_timeout(self):
        timeout = int(self.get_nodes_timeout() * 1.5)
        if timeout < JobConstant.INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MIN:
            timeout = JobConstant.INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MIN
        elif timeout > JobConstant.INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MAX:
            timeout = JobConstant.INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MAX

        return timeout

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

        # use 1.5 * rdzv-timeout with min: 600 and max: as timeout
        timeout = self._get_insufficient_timeout()
        logger.debug(
            f"Is training hang by insufficient worker with timeout: {timeout}."
        )

        nodes = self._get_nodes()
        cur_nodes = list(nodes.values())

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
        if (
            len(available_nodes) > 0
            and len(available_nodes) < self.get_min_nodes_required()
        ):
            if self._last_insufficient_nodes_timestamp == 0:
                self._last_insufficient_nodes_timestamp = int(now)
                logger.warning(
                    f"Job with insufficient nodes: {cur_nodes}. "
                    f"Need at least: {self.get_min_nodes_required()}."
                )
            else:
                if now - self._last_insufficient_nodes_timestamp > timeout:
                    logger.warning(
                        f"Job with insufficient nodes: {cur_nodes} "
                        f"lasts for more than {timeout}s. "
                        f"Need at least: {self.get_min_nodes_required()}."
                    )
                    return True
        else:
            self._last_insufficient_nodes_timestamp = 0

        return False

    def has_node_required_info(self):
        if (
            self._nodes_required[0]
            and self._nodes_required[1]
            and self._nodes_required[2]
        ):
            return True
        return False

    def get_min_nodes_required(self) -> int:
        """Notice: it is meaningless when the result is 0."""

        return self._nodes_required[0]

    def get_max_nodes_required(self) -> int:
        """Notice: it is meaningless when the result is 0."""

        return self._nodes_required[1]

    def get_nodes_timeout(self) -> int:
        """Notice: it is meaningless when the result is 0."""

        return self._nodes_required[2]

    def update_node_required_info(self, nodes_required: Tuple[int, int, int]):
        self._nodes_required = nodes_required

    def is_all_workers_node_check_failed(self):
        nodes = self._get_nodes()
        return all([node.is_node_check_failed() for _, node in nodes.items()])

    def is_all_initial_workers_node_check_failed(
        self, worker_num: int, min_worker_num: int = 4
    ):
        """
        Check all initial workers are check-failed(exclude new relaunched workers).

        Will skip judgement if worker number is less than min_worker_num because
        when the number of workers is too small (especially in single-machine
        or dual-machine scenarios), the probability of an all check failure
        increases significantly. At the same time, in scenarios with
        fewer workers, the number of fault tolerance attempts is limited by
        the total number of workers, which does not lead to excessively high
        fault tolerance costs. Hence, this boundary restriction is imposed.
        """

        if worker_num < min_worker_num:
            logger.debug(
                "Skip all initial workers node-check failed judgement "
                f"for worker num: {worker_num} < {min_worker_num}."
            )
            return False

        nodes = [
            node
            for _, node in self._get_nodes().items()
            if node.id < worker_num
        ]
        return len(nodes) > 0 and all(
            [node.is_node_check_failed() for node in nodes]
        )

    def is_all_workers_succeeded_exited(self):
        # check if all ranks already succeeded and exited
        succeeded_exited_ranks = set()
        nodes = self._get_nodes()
        actual_max_rank = max(node.rank_index for node in list(nodes.values()))

        for _, node in nodes.items():
            if node.is_succeeded_and_exited():
                succeeded_exited_ranks.add(node.rank_index)

        return len(succeeded_exited_ranks) == (actual_max_rank + 1)
