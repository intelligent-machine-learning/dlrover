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

import itertools
import math
import threading
import time
from abc import ABCMeta, abstractmethod
from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional

from dlrover.python.common.constants import (
    DistributionStrategy,
    JobConstant,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
    PriorityClass,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.scaler.base_scaler import ScalePlan
from dlrover.python.scheduler.job import JobArgs

_dlrover_context = Context.singleton_instance()

ALIVE_STATUS = [NodeStatus.INITIAL, NodeStatus.PENDING, NodeStatus.RUNNING]


def set_critical_node(
    job_nodes: Dict[str, Dict[int, Node]],
    ps_is_critical=True,
    ps_relaunch_max_num=0,
    critical_worker_index: Dict[int, int] = {},
):
    """Set critial nodes of a job.
    Args:
        job_nodes: nodes of a job.
        ps_is_critical: bool, whether all PS are critial.
        ps_relaunch_max_num: the maximum relaunch number of PS.
        critical_worker_index: a dict where the key is the index of critical
            workers and the value is the relaunchable count of the worker.
    """
    logger.info("Critical workers = %s", critical_worker_index)
    if NodeType.PS in job_nodes:
        for node in job_nodes[NodeType.PS].values():
            node.critical = ps_is_critical
            if node.critical:
                node.max_relaunch_count = ps_relaunch_max_num
    if NodeType.WORKER in job_nodes:
        for i, node in job_nodes[NodeType.WORKER].items():
            if node.id not in critical_worker_index:
                continue
            node.critical = True
            node.max_relaunch_count = critical_worker_index[i]
    if NodeType.EVALUATOR in job_nodes:
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.critical = True
    if NodeType.CHIEF in job_nodes:
        for node in job_nodes[NodeType.CHIEF].values():
            node.critical = True


def update_nodes_priority(job_nodes: Dict[str, Dict[int, Node]]):
    for nodes in job_nodes.values():
        group_node_num = len(nodes)
        for node in nodes.values():
            node.update_priority(group_node_num)


def get_critical_worker_index(params: JobArgs):
    """Get indices of critical workers.
    Returns:
        A dict: the key is the index of critical
            workers and the value is the relaunchable count of the worker.
    """
    critical_worker_index = {}
    worker_params = params.node_args[NodeType.WORKER]

    try:
        if worker_params.critical_nodes == "":
            # for default, worker0 is critical if PS strategy with
            # custom training
            if params.distribution_strategy == DistributionStrategy.PS:
                critical_worker_index[0] = worker_params.restart_count
        elif worker_params.critical_nodes == "all":
            for i in range(worker_params.group_resource.count):
                critical_worker_index[i] = worker_params.restart_count
        elif worker_params.critical_nodes != "none":
            for pod_relaunch_conf in worker_params.critical_nodes.split(","):
                # The conf is "pod_index:relaunch_times"
                pod_relaunch = pod_relaunch_conf.strip().split(":")
                critical_worker_index[int(pod_relaunch[0])] = int(
                    pod_relaunch[1]
                )

        return critical_worker_index
    except Exception as e:
        logger.warning(
            f"Invalid worker params: {worker_params.__dict__}, error: {str(e)}"
        )
        return {}


def get_pending_timeout():
    if _dlrover_context.seconds_to_wait_pending_pod <= 0:
        return JobConstant.PENDING_NODE_TIMEOUT_DEFAULT_MIN
    return _dlrover_context.seconds_to_wait_pending_pod


def reduce_timeout_pending_node_resource(node: Node):
    """Reduce CPU cores or memory and relaunch it if the pending
    time is too long"""
    now = time.time()
    if (
        node.is_released
        or not node.create_time
        or node.config_resource.gpu_num > 0
    ):
        return False
    pending_time = now - node.create_time.timestamp()
    pending_timeout = get_pending_timeout()
    if pending_time < pending_timeout:
        return False

    original_cpu = node.config_resource.cpu
    new_cpu = math.ceil(
        original_cpu / _dlrover_context.factor_to_cut_pending_cpu
    )
    if new_cpu > NodeResourceLimit.MIN_CPU_CORES:
        node.config_resource.cpu = new_cpu
        logger.info(
            "Pod %s pending time %s beyonds %s.Delete and relaunch it with CPU %s",
            node.name,
            pending_time,
            pending_timeout,
            new_cpu,
        )
    original_memory = node.config_resource.memory
    new_memory = math.ceil(
        original_memory / _dlrover_context.factor_to_cut_pending_mem
    )
    if new_memory > NodeResourceLimit.MIN_MEMORY:
        node.config_resource.memory = new_memory
        logger.info(
            "Pod %s pending time %s beyonds %s.Delete and relaunch it with memory %s",
            node.name,
            pending_time,
            pending_timeout,
            new_memory,
        )
    return True


def skip_pending_judgement(strategy) -> bool:
    return strategy == 0


def is_key_nodes_pending_judgement(strategy) -> bool:
    return strategy == 1


def is_all_nodes_pending_judgement(strategy) -> bool:
    return strategy == 2


class TrainingNodeManager(object):
    def __init__(
        self,
        node_type: str,
        new_node_name_fn,
    ):
        """
        Args:
            node_type: node type
            new_node_name_fn: new node name function
        """
        self._job_context = get_job_context()
        self._node_type = node_type
        self._new_node_name_fn = new_node_name_fn
        self._lock = threading.Lock()
        self._node_id_iter = None
        self._node_rank_iter = None
        self.update_nodes_iter()

    @property
    def cur_nodes(self):
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        cur_nodes = [node.name for node in nodes.values()]
        return cur_nodes

    def _get_mutable_nodes(self):
        """
        The object(original) got from this method can be updated directly.
        """

        return self._job_context.get_mutable_job_nodes(self._node_type)

    def _get_nodes(self):
        """
        The object(copied) got from this method can be read only.
        """

        return self._job_context.job_nodes_by_type(self._node_type)

    def _update_node(self, node: Node):
        self._job_context.update_job_node(node)
        if node.has_group():
            self._job_context.update_job_node_by_group(node)

    def update_nodes_iter(self, update_rank_iter=True):
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        self._node_id_iter = itertools.count(
            max(nodes.keys()) + 1 if len(nodes) > 0 else 0
        )
        if update_rank_iter:
            self._node_rank_iter = itertools.count(len(nodes))

    def get_next_node_id(self):
        self.update_nodes_iter(update_rank_iter=False)
        return next(self._node_id_iter)

    def remove_node(self, node_id):
        plan = ScalePlan()
        with self._lock:
            worker = self._job_context.job_node(self._node_type, node_id)
            if worker is None:
                logger.info("Delete non-existed worker %s", node_id)
                return plan
            if worker.status in [NodeStatus.DELETED, NodeStatus.INITIAL]:
                logger.error("Unknown deletable worker id: %s" % node_id)
                return
        worker.is_released = True
        self._update_node(worker)
        plan.remove_nodes.append(worker)
        return plan

    def relaunch_node(self, node: Node, remove_exited_node=False):
        plan = ScalePlan()
        with self._lock:
            new_id = self.get_next_node_id()
            new_name = self._new_node_name_fn(node.type, new_id)
            relaunch_node = node.generate_relaunch_node(new_id, new_name)
            self._update_node(relaunch_node)
        logger.info(
            f"Relaunch node {node.name} to {new_id} with relaunch count: "
            f"{node.relaunch_count}/{node.max_relaunch_count}"
        )
        plan.launch_nodes.append(relaunch_node)
        if remove_exited_node and not node.is_released and node.exited():
            node.is_released = True
            self._update_node(node)
            plan.remove_nodes.append(node)
        return plan

    def relaunch_nodes(self, nodes: List[Node], remove_exited_node=False):
        plan = ScalePlan()

        for node in nodes:
            sub_plan = self.relaunch_node(node, remove_exited_node)
            plan.launch_nodes.extend(sub_plan.launch_nodes)
            plan.remove_nodes.extend(sub_plan.remove_nodes)

        return plan

    def reduce_pending_node_resource(self):
        """Cut down CPU cores of pending PS Pods"""
        plan = ScalePlan()

        # Avoid dictionary changed size during iteration.
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        cur_nodes = list(nodes.values())
        for node in cur_nodes:
            if node.status == NodeStatus.PENDING:
                reduced = reduce_timeout_pending_node_resource(node)
                if reduced:
                    node.relaunchable = False
                    self._update_node(node)
                    node_plan = self.relaunch_node(node)
                    plan.remove_nodes.append(node)
                    plan.merge(node_plan)
        return plan

    def get_pending_timeout_oom_recovered_node(self):
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        cur_nodes = list(nodes.values())
        now = time.time()
        nodes = []
        for node in cur_nodes:
            if (
                node.is_released
                or not node.create_time
                or node.status != NodeStatus.PENDING
            ):
                continue
            pending_time = now - node.create_time.timestamp()
            pending_timeout = get_pending_timeout()
            if node.is_recovered_oom and pending_time > pending_timeout:
                logger.info(
                    f"Node {node.name} with resource f{node.config_resource} "
                    f"and pends f{pending_time}s."
                )
                nodes.append(node)
        return nodes

    def get_running_nodes(self):
        """TensorFlow Chief nodes"""
        nodes = []
        with self._lock:
            training_nodes = self._job_context.job_nodes_by_type(
                self._node_type
            )
            for node in training_nodes.values():
                if node.status == NodeStatus.RUNNING and not node.is_released:
                    nodes.append(node)
        return nodes

    def all_nodes_exited(self):
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        if len(nodes) == 0:
            logger.debug(f"All {self._node_type} nodes exited")
            return True
        counter = self._get_node_counter()

        # At start, there may be no launched worker.
        if len(counter) == 1 and NodeStatus.INITIAL in counter:
            return False

        with self._lock:
            high_worker_num = 0
            running_workers = []
            pending_high_workers = []
            pending_low_workers = []
            for worker_id, worker in nodes.items():
                if worker.is_released:
                    continue
                if worker.config_resource.priority == PriorityClass.LOW:
                    if worker.status == NodeStatus.RUNNING:
                        running_workers.append(worker_id)
                    elif worker.status in [
                        NodeStatus.INITIAL,
                        NodeStatus.PENDING,
                    ]:
                        pending_low_workers.append(worker_id)
                else:
                    high_worker_num += 1
                    if worker.status == NodeStatus.RUNNING:
                        running_workers.append(worker_id)
                    elif worker.status in [
                        NodeStatus.INITIAL,
                        NodeStatus.PENDING,
                    ]:
                        pending_high_workers.append(worker_id)

            logger.debug(
                f"Check all {self._node_type} nodes exited: "
                f"{running_workers} {pending_high_workers} "
                f"{pending_low_workers} {high_worker_num}"
            )
            if (
                running_workers
                or pending_high_workers
                or (not high_worker_num and pending_low_workers)
            ):
                return False
            else:
                return True

    def all_nodes_deleted(self):
        counter = self._get_node_counter()
        all_deleted = all([status == NodeStatus.DELETED for status in counter])
        return all_deleted

    def all_nodes_failed(self):
        counter = self._get_node_counter()
        if len(counter) == 1 and NodeStatus.INITIAL in counter:
            return False

        all_failed = all(
            [
                status
                in [NodeStatus.FAILED, NodeStatus.DELETED, NodeStatus.INITIAL]
                for status in counter
            ]
        )
        return all_failed

    def running_nodes_hanged(self) -> List[bool]:
        cur_time = time.time()
        node_hang = []
        nodes_dict = self._job_context.job_nodes_by_type(self._node_type)
        nodes = list(nodes_dict.values())  # Avoid dictionary changed size.
        for node in nodes:
            if node.status == NodeStatus.RUNNING:
                timeout = NodeResourceLimit.MAX_HANG_TIMEOUT_SECS
                hang = (
                    node.start_hang_time > 0
                    and cur_time - node.start_hang_time > timeout
                )
                if not node.hang and hang:
                    time_array = time.localtime(node.start_hang_time)
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
                    logger.warning(
                        f"Node {node.name} hangs with timeout "
                        f"{timeout} from {date_time}!!!"
                    )
                node.hang = hang
                self._update_node(node)
                node_hang.append(hang)
        return node_hang

    def _get_node_counter(self):
        with self._lock:
            nodes = self._job_context.job_nodes_by_type(self._node_type)
            return Counter([node.status for node in nodes.values()])

    def update_critical_node(self, critical_node_restarts):
        """Update critical node by a dict.
        Args:
            critical_node_restarts: A dict where keys are node ids
                and values are the relaunchable number of nodes
        """
        logger.info("Update critical worker {}".format(critical_node_restarts))
        nodes = self._job_context.job_nodes_by_type(self._node_type)
        for id, node in nodes.items():
            if id in critical_node_restarts:
                node.critical = True
                node.max_relaunch_count = critical_node_restarts[id]

    def _get_pending_timeout(self):
        timeout = _dlrover_context.seconds_to_wait_pending_pod
        if timeout <= 0:
            return 0
        if timeout < JobConstant.PENDING_NODE_TIMEOUT_DEFAULT_MIN:
            timeout = JobConstant.PENDING_NODE_TIMEOUT_DEFAULT_MIN

        return timeout

    def _get_group_pending_timeout(self):
        timeout = _dlrover_context.seconds_to_wait_group_pending_pod
        if timeout <= 0:
            return 0
        if timeout < JobConstant.GROUP_PENDING_NODE_TIMEOUT_DEFAULT_MIN:
            timeout = JobConstant.GROUP_PENDING_NODE_TIMEOUT_DEFAULT_MIN

        return timeout


@dataclass
class SyncNodeTrainingPorts:
    training_port: int = 0
    next_check_port: int = 0


class ExternalConfig(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_elastic_run_configs(self) -> Dict[str, str]:
        pass


class TrainingNodeConfig:
    def __init__(self, external_config: Optional[ExternalConfig] = None):
        self._lock = Lock()
        self._recv_node_training_ports: Dict[int, int] = {}
        self._node_training_port = 0
        self._next_check_node_training_port = 0
        self._n_node = 0
        self._external_config = external_config

    def set_node_num(self, num):
        logger.info(f"set worker count: {num}")
        self._n_node = num

    def sync_node_training_port(self, node_id, port) -> SyncNodeTrainingPorts:
        with self._lock:
            # port is already decided
            if self._node_training_port > 0:
                return SyncNodeTrainingPorts(
                    training_port=self._node_training_port, next_check_port=0
                )
            # workers should start next round port check
            if port < self._next_check_node_training_port:
                return SyncNodeTrainingPorts(
                    training_port=0,
                    next_check_port=self._next_check_node_training_port,
                )
            self._recv_node_training_ports[node_id] = port
            logger.info(f"recv ports: {self._recv_node_training_ports.keys()}")
            if len(self._recv_node_training_ports) == self._n_node:
                min_port = 0
                max_port = 0
                for _, recv_port in self._recv_node_training_ports.items():
                    if min_port == 0 or min_port > recv_port:
                        min_port = recv_port
                    if max_port < recv_port:
                        max_port = recv_port
                if min_port != max_port:
                    self._recv_node_training_ports.clear()
                    self._next_check_node_training_port = max_port
                    logger.info(
                        f"fail to sync node training ports: "
                        f"{self._recv_node_training_ports}, "
                        f"next sync port: {max_port}"
                    )
                    return SyncNodeTrainingPorts(
                        training_port=0,
                        next_check_port=self._next_check_node_training_port,
                    )
                else:
                    self._node_training_port = max_port
                    return SyncNodeTrainingPorts(
                        training_port=self._node_training_port,
                        next_check_port=0,
                    )
            else:
                return SyncNodeTrainingPorts(
                    training_port=0, next_check_port=0
                )

    def get_elastic_run_configs(self) -> Dict[str, str]:
        if not self._external_config:
            logger.warning("External config not set")
            return {}
        return self._external_config.get_elastic_run_configs()
