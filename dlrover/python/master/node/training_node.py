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
import math
import threading
import time
from collections import Counter
from typing import Dict, List

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
    PriorityClass,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
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

    if worker_params.critical_nodes == "":
        # for default, worker0 is critical if PS strategy with custom training
        if params.distribution_strategy == DistributionStrategy.PS:
            critical_worker_index[0] = worker_params.restart_count
    elif worker_params.critical_nodes == "all":
        for i in range(worker_params.group_resource.count):
            critical_worker_index[i] = worker_params.restart_count
    elif worker_params.critical_nodes != "none":
        for pod_relaunch_conf in worker_params.critical_nodes.split(","):
            # The conf is "pod_index:relaunch_times"
            pod_relaunch = pod_relaunch_conf.strip().split(":")
            critical_worker_index[int(pod_relaunch[0])] = int(pod_relaunch[1])

    return critical_worker_index


def cut_timeout_pending_node_cpu(node: Node):
    """Cut down CPU cores and relaunch it if the pending
    time is too long"""
    now = time.time()
    if node.is_released or not node.create_time:
        return False
    pending_time = now - node.create_time.timestamp()
    if pending_time < _dlrover_context.seconds_to_wait_pending_pod:
        return False

    original_cpu = node.config_resource.cpu
    new_cpu = math.ceil(
        original_cpu / _dlrover_context.factor_to_cut_pending_cpu
    )
    if new_cpu > NodeResourceLimit.MIN_CPU_CORES:
        node.config_resource.cpu = new_cpu
        logger.info(
            "Pod %s pending time %s beyonds %s."
            "Delete and relaunch it with CPU %s",
            node.name,
            pending_time,
            _dlrover_context.seconds_to_wait_pending_pod,
            new_cpu,
        )
        return True
    else:
        return False


class TrainingNodeManager(object):
    def __init__(
        self,
        nodes: Dict[int, Node],
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
        self._nodes = nodes
        self._new_node_name_fn = new_node_name_fn
        self._lock = threading.Lock()
        self._node_id_iter = itertools.count(len(self._nodes))
        self._rank_id_iter = itertools.count(len(self._nodes))

    def update_nodes(self, nodes):
        self._nodes = nodes
        self._node_id_iter = itertools.count(len(self._nodes))
        self._rank_id_iter = itertools.count(len(self._nodes))

    def remove_node(self, node_id):
        plan = ScalePlan()
        if node_id not in self._nodes:
            logger.info("Delete non-existed worker %s", node_id)
            return plan
        worker = self._nodes[node_id]
        with self._lock:
            if worker.status in [NodeStatus.DELETED, NodeStatus.INITIAL]:
                logger.error("Unknown deletable worker id: %s" % node_id)
                return
        worker.is_released = True
        plan.remove_nodes.append(worker)
        return plan

    def relaunch_node(self, node: Node):
        plan = ScalePlan()
        with self._lock:
            node.is_released = True
            new_id = next(self._node_id_iter)
            relaunch_node = node.get_relaunch_node_info(new_id)
            self._nodes[new_id] = relaunch_node
        logger.info("Relaunch node %s to %s", node.name, new_id)
        plan.launch_nodes.append(
            Node(
                node.type,
                new_id,
                copy.deepcopy(relaunch_node.config_resource),
                rank_index=node.rank_index,
                name=self._new_node_name_fn(node.type, new_id),
                service_addr=node.service_addr,
                relaunch_count=relaunch_node.relaunch_count,
            )
        )
        return plan

    def cut_pending_node_cpu(self):
        """Cut down CPU cores of pendding PS Pods"""
        plan = ScalePlan()
        nodes = copy.deepcopy(self._nodes)
        for node in nodes.values():
            if node.status == NodeStatus.PENDING:
                cut_cpu = cut_timeout_pending_node_cpu(node)
                if cut_cpu:
                    node.relaunchable = False
                    node_plan = self.relaunch_node(node)
                    plan.merge(node_plan)
        return plan

    def get_running_nodes(self):
        """TensorFlow Chief nodes"""
        nodes = []
        with self._lock:
            for node in self._nodes.values():
                if node.status == NodeStatus.RUNNING:
                    nodes.append(node)
        return nodes

    def all_nodes_exited(self):
        if len(self._nodes) == 0:
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
            for worker_id, worker in self._nodes.items():
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
        for _, node in self._nodes.items():
            if node.status == NodeStatus.RUNNING:
                hang = (
                    node.start_hang_time > 0
                    and cur_time - node.start_hang_time
                    > NodeResourceLimit.MAX_HANG_TIMEOUT_SECS
                )
                if hang:
                    time_array = time.localtime(node.start_hang_time)
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
                    logger.warning("Node %s hangs at %s", node.name, date_time)
                node_hang.append(hang)
        return node_hang

    def _get_node_counter(self):
        with self._lock:
            return Counter([node.status for node in self._nodes.values()])

    def update_critical_node(self, critical_node_restarts):
        """Update critical node by a dict.
        Args:
            critical_node_restarts: A dict where keys are node ids
                and values are the relaunchable number of nodes
        """
        logger.info("Update critical worker {}".format(critical_node_restarts))
        for id, node in self._nodes.items():
            if id in critical_node_restarts:
                node.critical = True
                node.max_relaunch_count = critical_node_restarts[id]
