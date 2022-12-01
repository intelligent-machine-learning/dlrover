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
from typing import Dict

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.master.scaler.base_scaler import ScalePlan, LaunchNode
from dlrover.python.master.watcher.base_watcher import Node

_dlrover_context = Context.instance()

ALIVE_STATUS = [NodeStatus.INITIAL, NodeStatus.PENDING, NodeStatus.RUNNING]


def set_critical_node(
    job_nodes: Dict[str, Dict[int, Node]],
    ps_is_critical=True,
    critical_worker_index={},
    ps_relaunch_max_num=0,
):
    """
    pod_info is a dict, where pod_info[type][id] is a PodInfo instance
    Set is_critical_pod values accordingly
    """
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


def get_critical_worker_index(args):
    critical_worker_index = {}

    if args.critical_worker_index == "default":
        # for default, worker0 is critical if PS strategy with custom training
        if args.distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            critical_worker_index[0] = args.relaunch_on_worker_failure
    elif args.critical_worker_index == "all":
        for i in range(args.num_workers):
            critical_worker_index[i] = args.relaunch_on_worker_failure
    elif args.critical_worker_index != "none":
        for pod_relaunch_conf in args.critical_worker_index.split(","):
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
        self._lock = threading.Lock()
        self._node_id_iter = itertools.count(len(self._nodes))

    def update_nodes(self, nodes):
        self._nodes = nodes
        self._node_id_iter = itertools.count(len(self._nodes))

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
        plan.remove_nodes.append(worker.name)
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
            LaunchNode(
                node.type,
                new_id,
                node.task_index,
                relaunch_node.config_resource,
            )
        )
        plan.remove_nodes.append(node.name)
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
        counter = self._get_node_counter()

        # At start, there may be no launched worker.
        if len(counter) == 1 and NodeStatus.INITIAL in counter:
            return False

        all_exited = True
        with self._lock:
            for node in self._nodes.values():
                if not node.is_released and (node.status in ALIVE_STATUS):
                    all_exited = False
                    break
        return all_exited

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

    def _get_node_counter(self):
        with self._lock:
            return Counter([node.status for node in self._nodes.values()])
