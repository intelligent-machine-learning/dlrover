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

import collections
import copy
import itertools
import threading
from typing import Dict, List

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.training_node import TrainingNodeManager
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.scaler.base_scaler import ScalePlan


class ParameterServerManager(TrainingNodeManager):
    def __init__(
        self,
        ps_nodes: Dict[int, Node],
        job_resource: JobResource,
        max_relaunch_num,
        new_service_fn,
        new_node_name_fn,
    ):
        """
        Args:
            ps_nodes: A dictionary where the key is the index of PS pod
                and the value is the PodInfo instance of PS pod.
            job_resource: the resource configuration of a job.
            max_relaunch_num: The maximum relaunch number of PS.
            new_service_fn: A callable function to generate a server name of
                PS.
            new_node_name_fn: A callable function to generate a node name of
                PS.
        """
        super(ParameterServerManager, self).__init__(
            ps_nodes, new_node_name_fn
        )
        self._max_relaunch_num = max_relaunch_num
        self._job_resource = job_resource
        self._new_service_fn = new_service_fn
        self._pre_dropped_ps: List[Node] = []
        self._lock = threading.Lock()
        self._ready_for_new_ps_cluster = False
        self._migrated_ps_nodes: Dict[int, Node] = {}
        self._next_training_ps_cluster: List[Node] = []
        self._training_ps_cluster: List[Node] = []
        self._node_id_iter = itertools.count(self._job_resource.ps_num)
        self._init_training_ps_cluster()

    def _init_training_ps_cluster(self):
        for node in self._nodes.values():
            alive = node.status in [
                NodeStatus.INITIAL,
                NodeStatus.PENDING,
                NodeStatus.RUNNING,
            ]
            logger.info("PS : %s", node)
            if (
                node.id not in self._migrated_ps_nodes
                and not node.is_released
                and alive
            ):
                self._training_ps_cluster.append(node)
                self._next_training_ps_cluster.append(node)

    def relaunch_node(self, node: Node):
        plan = ScalePlan()
        with self._lock:
            node.is_released = True
            new_id = next(self._node_id_iter)
            self._nodes[new_id] = node.get_relaunch_node_info(new_id)
            if node in self._training_ps_cluster:
                i = self._training_ps_cluster.index(node)
                self._training_ps_cluster[i] = self._nodes[new_id]
        logger.info("Relaunch node %s to %s", node.name, new_id)
        plan.launch_nodes.append(
            Node(
                node.type,
                new_id,
                copy.deepcopy(node.config_resource),
                rank_index=node.rank_index,
                name=self._new_node_name_fn(node.type, new_id),
                service_addr=node.service_addr,
                relaunch_count=node.relaunch_count,
            )
        )
        return plan

    def adjust_ps(self, ps_resource: NodeGroupResource):
        logger.info(
            "Adjust ps resource to num = %s, cpu = %s, memory = %sMi",
            ps_resource.count,
            ps_resource.node_resource.cpu,
            ps_resource.node_resource.memory,
        )
        plan = ScalePlan()
        alive_num = len(self.get_training_ps_cluster())
        logger.info("The current number of alive PS is %s", alive_num)
        if ps_resource.count > alive_num:
            new_ps = self._scale_up_ps(ps_resource.count - alive_num)
            plan.launch_nodes.extend(new_ps)
        elif ps_resource.count < alive_num:
            self._scale_down_ps(alive_num - ps_resource.count)
        return plan

    def _scale_up_ps(self, up_num):
        logger.info("Scale up ps with the number %s", up_num)
        new_ps = []
        with self._lock:
            self._ready_for_new_ps_cluster = False
            alive_num = len(self.get_training_ps_cluster())
            task_id_iter = itertools.count(alive_num)
            for _ in range(up_num):
                ps_id = next(self._node_id_iter)
                task_id = next(task_id_iter)
                service_addr = self._new_service_fn(NodeType.PS, ps_id)
                ps_resource = self._job_resource.get_node_group_resource(
                    NodeType.PS
                ).node_resource
                ps = Node(
                    NodeType.PS,
                    node_id=ps_id,
                    rank_index=task_id,
                    name=self._new_node_name_fn(NodeType.PS, ps_id),
                    max_relaunch_count=self._max_relaunch_num,
                    config_resource=copy.deepcopy(ps_resource),
                    critical=True,
                    service_addr=service_addr,
                )
                self._nodes[ps_id] = ps
                new_ps.append(ps)
                logger.info("Create PS %s", ps)
        return new_ps

    def _scale_down_ps(self, down_num):
        with self._lock:
            self._pre_dropped_ps = []
            self._ready_for_new_ps_cluster = False
            new_ps_num = self._job_resource.ps_num - down_num
            self._job_resource.update_node_group_resource(
                NodeType.PS, new_ps_num, 0, 0
            )
            running_ps = self._get_alive_ps()
            for node in reversed(running_ps):
                if down_num <= 0:
                    break
                self._pre_dropped_ps.append(node)
                down_num -= 1
        dropped_ps = [ps.name for ps in self._pre_dropped_ps]
        logger.info("Scale down PS %s", dropped_ps)

    def process_after_ps_cluster_ready(self):
        self._ready_for_new_ps_cluster = True
        self._training_ps_cluster = []
        logger.info("Process PS nodes after ps training is ready")
        self._training_ps_cluster.extend(self._next_training_ps_cluster)
        logger.info(
            "Update training PS cluster = %s", self._training_ps_cluster
        )
        plan = ScalePlan()
        with self._lock:
            while self._pre_dropped_ps:
                node = self._pre_dropped_ps.pop()
                node.critical = False
                node.relaunchable = False
                node.is_released = True
                if node.id in self._migrated_ps_nodes:
                    self._migrated_ps_nodes.pop(node.id)
                plan.remove_nodes.append(node)
        return plan

    def _get_alive_ps(self) -> List[Node]:
        """Get all running PS pods"""
        alive_ps = []
        for node in self._nodes.values():
            if node.status == NodeStatus.RUNNING and not node.is_released:
                alive_ps.append(node)
        return alive_ps

    def get_next_training_ps_cluster(self):
        """Get the next training PS cluster.
        After rescaling PS, it should return the new PS set until
        all new PS are running. Otherwise, it returns the old PS set.
        """
        if self._ready_for_new_ps_cluster:
            return self._next_training_ps_cluster

        all_new_ps_ready = True
        for node in self._nodes.values():
            if not node.is_released and node.status in [
                NodeStatus.INITIAL,
                NodeStatus.PENDING,
            ]:
                all_new_ps_ready = False
                break
        if all_new_ps_ready:
            self._next_training_ps_cluster = self._get_all_non_migrated_ps()
        return self._next_training_ps_cluster

    def _get_all_non_migrated_ps(self):
        """Get all running PS pods without migrated PS nodes for training"""
        training_ps = {}
        with self._lock:
            alive_ps = self._get_alive_ps()
            self._pre_drop_migrated_ps(alive_ps)
            for ps in alive_ps:
                if ps not in self._pre_dropped_ps:
                    training_ps[ps.rank_index] = ps
        training_ps = collections.OrderedDict(sorted(training_ps.items()))
        return list(training_ps.values())

    def _pre_drop_migrated_ps(self, alive_ps: List[Node]):
        """Move the old ps into pre-dropped list and
        Kill those old pods after the new training cluster is built.
        """
        for node in self._migrated_ps_nodes.values():
            if node.status != NodeStatus.RUNNING:
                return
        for node in alive_ps:
            if (
                node.id in self._migrated_ps_nodes
                and node.status == NodeStatus.RUNNING
            ):
                if node not in self._pre_dropped_ps:
                    self._pre_dropped_ps.append(node)

    def get_total_request_cpu(self):
        total_cpu = 0
        for node in self._get_alive_ps():
            total_cpu += node.config_resource.cpu
        return total_cpu

    def get_training_ps_cluster(self):
        """Get the ps nodes who are training."""
        if not self._training_ps_cluster:
            self._init_training_ps_cluster()
        training_ps: List[Node] = []
        for ps in self._training_ps_cluster:
            if not ps.is_released and ps.status != NodeStatus.FAILED:
                training_ps.append(ps)
        logger.info("training_ps_cluster is {}".format(training_ps))
        return training_ps

    def get_ready_for_new_ps_cluster(self):
        return self._ready_for_new_ps_cluster

    def get_ps_addrs(self):
        """Get the address list of ps services"""
        ps_addrs = {}
        for ps in list(self._nodes.values()):
            if (
                ps.id not in self._migrated_ps_nodes
                and not ps.is_released
                and ps.status
                in [NodeStatus.INITIAL, NodeStatus.PENDING, NodeStatus.RUNNING]
            ):
                ps_addrs[ps.rank_index] = ps.service_addr
        ps_addrs = collections.OrderedDict(sorted(ps_addrs.items()))
        return list(ps_addrs.values())

    def delete_running_ps(self):
        """Delete all running ps pods"""
        plan = ScalePlan()
        for node in list(self._nodes.values()):
            if (
                node.status in [NodeStatus.RUNNING, NodeStatus.PENDING]
                and not node.is_released
            ):
                node.critical = False
                node.relaunchable = False
                logger.info(
                    "Remove the pod {} after the worker-0 completed".format(
                        node.name
                    )
                )
                node.is_released = True
                node.status = NodeStatus.DELETED
                plan.remove_nodes.append(node)
        return plan

    def migrate_parameter_servers(self, ps_nodes: Dict[str, NodeResource]):
        plan = ScalePlan()
        for name, resource in ps_nodes.items():
            node = self._migrate_parameter_server(
                name, resource.cpu, resource.memory
            )
            if node:
                plan.launch_nodes.append(copy.deepcopy(node))
        return plan

    def _migrate_parameter_server(self, name: str, cpu=0, memory=0):
        """Migrate the parameter server node into a new pod"""
        old_ps_id = int(name.split("-")[-1])
        original_pod = self._nodes[old_ps_id]
        if (
            old_ps_id in self._migrated_ps_nodes
            or original_pod.is_released
            or original_pod.status
            not in [NodeStatus.PENDING, NodeStatus.RUNNING]
        ):
            return

        resource = copy.deepcopy(original_pod.config_resource)
        with self._lock:
            self._ready_for_new_ps_cluster = False
            new_ps_id = next(self._node_id_iter)
            resource.cpu = cpu
            resource.memory = memory
            logger.info(
                "resource memory = %s, cpu = %s", resource.memory, resource.cpu
            )

            service_addr = self._new_service_fn(NodeType.PS, new_ps_id)
            new_node = Node(
                NodeType.PS,
                node_id=new_ps_id,
                rank_index=original_pod.rank_index,
                max_relaunch_count=self._max_relaunch_num,
                config_resource=resource,
                critical=True,
                service_addr=service_addr,
                name=self._new_node_name_fn(NodeType.PS, new_ps_id),
            )
            self._nodes[new_ps_id] = new_node
            self._migrated_ps_nodes[old_ps_id] = new_node
            logger.info("Migrated PS %s to PS %s", old_ps_id, new_ps_id)
            return new_node

    def exist_migrated_ps_nodes(self):
        return len(self._migrated_ps_nodes) > 0

    def is_all_running(self):
        running_ps = [
            pod_info.id
            for pod_info in self._nodes.values()
            if pod_info.status == NodeStatus.RUNNING
        ]
        return len(running_ps) == self._job_resource.ps_num
