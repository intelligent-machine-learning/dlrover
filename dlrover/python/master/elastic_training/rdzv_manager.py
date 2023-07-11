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

import math
import time
from abc import ABCMeta, abstractmethod
from threading import Lock
from typing import Dict, List

from dlrover.python.common.constants import (
    NetworkFailureReason,
    RendezvousName,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


class RendezvousParameters(object):
    """Holds the parameters to construct rendezvous.
    Args:
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        waiting_timeout:
            An additional wait amount before completing the rendezvous once
            the rendezvous has the minimum number of required participants.
            Default 30s,
    """

    def __init__(
        self,
        min_nodes: int,
        max_nodes: int,
        waiting_timeout=30,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.waiting_timeout = waiting_timeout


class RendezvousManager(metaclass=ABCMeta):
    def __init__(self):
        self._lock = Lock()
        self._alive_nodes = set()
        self._released_workers = []
        self._waiting_nodes: Dict[int, int] = {}
        self._rdzv_nodes = {}
        self._lastcall_time = 0
        self._rdzv_params = RendezvousParameters(0, 0)
        self._rdzv_round = 0
        self._node_unit = 1
        self._name = ""
        self._latest_rdzv_nodes = []

    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        self._alive_nodes.add(node.id)
        logger.info(
            f"Add alive worker {node.name} to {self._name} rendezvous."
        )

    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        if node.id in self._alive_nodes:
            self._alive_nodes.remove(node.id)
            logger.info(
                f"Remove exited worker {node.name} from "
                f"{self._name} rendezvous."
            )
            self._waiting_nodes.pop(node.rank_index, 0)

    def update_rdzv_params(
        self, min_nodes, max_ndoes, waiting_timeout, node_unit
    ):
        """Update rendezvous parameters
        Args:
            min_nodes: The minimum number of nodes.
            max_nodes: THe maximum number of nodes.
            waiting_timeout: the time to wait more workers.
            worker_unit: the number unit of workers to build the communication
                world. This is, the number of nodes in a world should be
                a multiple of worker_unit.
        """
        self._rdzv_params.min_nodes = min_nodes
        self._rdzv_params.max_nodes = max_ndoes
        self._rdzv_params.waiting_timeout = waiting_timeout
        self._node_unit = node_unit

    def _check_rdzv_completed(self):
        rdzv_completed = False
        waiting_num = len(self._waiting_nodes)
        if len(self._waiting_nodes) == self._rdzv_params.max_nodes:
            rdzv_completed = True
        else:
            waiting_time = time.time() - self._lastcall_time
            if (
                waiting_num >= self._rdzv_params.min_nodes
                and waiting_time >= self._rdzv_params.waiting_timeout
            ):
                rdzv_completed = True
                waiting_num = (
                    waiting_num // self._node_unit
                ) * self._node_unit

        if rdzv_completed:
            node_ids = sorted(self._waiting_nodes.keys())[0:waiting_num]
            self._rdzv_nodes = {}
            for i in node_ids:
                self._rdzv_nodes[i] = self._waiting_nodes[i]
            self._latest_rdzv_nodes = list(self._rdzv_nodes.keys())
            self._waiting_nodes = dict()
            self._lastcall_time = 0
            logger.info(
                f"Completed {self._rdzv_round} round "
                f"rendezvous of elastic training is {self._rdzv_nodes}"
            )
        return rdzv_completed

    def not_joined_rdzv_nodes(self):
        """Return workers which do not join a rendezvous."""
        nodes = []
        if self._rdzv_nodes:
            for node_id in self._alive_nodes:
                if node_id not in self._rdzv_nodes:
                    nodes.append(node_id)
        return nodes

    def join_rendezvous(
        self,
        rank_id,
        local_world_size,
    ):
        """The node joins the current rond rendezvous.
        Args:
            node_id: the node ID which is unique in an ElasticJob of DLrover.
            local_world_size: the local world size of a node.

        Returns:
            int: the number of rendezvous round.
        """
        with self._lock:
            if rank_id in self._waiting_nodes:
                return
            self._waiting_nodes[rank_id] = local_world_size
            logger.info(f"{self._name} waiting nodes: {self._waiting_nodes}")
            self._rdzv_nodes = {}
            self._lastcall_time = time.time()
        return self._rdzv_round

    def num_nodes_waiting(self):
        """The elastic agent will restart training processes if it
        find the number of waiting nodes is not zero. The manager
        will notify all nodes to restart training processes immediately if
        ab existing node re-joins the next round rendezvous.
        If there are new nodes, the master notifies all nodes to re-join
        the next round rendezvous only when the number of waiting nodes
        is bigger than the number unit of nodes.
        """
        with self._lock:
            if self._has_node_restart():
                return len(self._waiting_nodes)
            elif len(self._waiting_nodes) >= self._node_unit:
                return len(self._waiting_nodes)
            return 0

    def _has_node_restart(self):
        """The node will restart training processes if it
        re-joins the rendezvous."""
        for rank_id in self._waiting_nodes.keys():
            if rank_id in self._latest_rdzv_nodes:
                return True
        return False

    @abstractmethod
    def get_comm_world(self, rank_id):
        """Get communication world of all alive nodes."""
        pass

    @abstractmethod
    def report_network_check_result(self, node_id: int, normal: bool):
        """The node updates its status"""
        pass


class ElasticTrainingRendezvousManager(RendezvousManager):
    """ElasticTrainingRendezvousManager runs on the DLRover master. The manager
    add workers into a waiting list and completes a rendezvous
    if the number of workers in the wait list is beyond the minimum
    nodes.

    The node report its ID and local_world_size to the manager.
    The manager will add the node into a waiting list to join the rendezvous
    and freeze the rendezvous if the size of waiting list is equal
    the max nodes or is bigger than the min nodes. Then the node will
    periodically query the world which contains
    all nodes like {0: 8, 1: 8, 2:8}. The key in the world dictionary
    is the node ID and the value is the local world size. In an
    Elasticjob of DLRover, the node has an unique node ID.
    """

    def __init__(self):
        super().__init__()
        self._name = RendezvousName.ELASTIC_TRAINING

    def get_comm_world(self, rank_id):
        """Return the communication world if a round rendezvous is completed.
        The rendezvous is completed if one of the following conditions
        is satisfied:
        1. The size of waiting node list is equal to the max_nodes.
        2. The size of waiting node list is bigger than the min_nodes and
            equal to the size of alive node list. What's more, no more worker
            join the rendezvous in waiting_timeout.

        Returns:
            world: Dict like {0: 8, 1: 8, 2: 8} where the key is the node ID
            and the value is the local world size of the node.
        """
        with self._lock:
            if not self._rdzv_nodes:
                rdzv_completed = self._check_rdzv_completed()
                if rdzv_completed:
                    self._rdzv_round += 1

            if rank_id not in self._rdzv_nodes:
                return self._rdzv_round, {}
            return self._rdzv_round, self._rdzv_nodes

    def report_network_check_result(self, node_id, normal):
        return


class NetworkCheckRendezvousManager(RendezvousManager):
    """NcclCheckRendezvousManager runs on the DLRover master. The task
    to check network contains 3 round to execute allgather on all nodes.
    We show the detail to check network assuming there are 4 nodes.
    Round 1: all nodes join a communication world {0:8, 1:8, 2:8, 3:8}
        where the key is the node id and the value is the local world size
        of the node. The check passes if allgather of all nodes is succeed.
        Otherwise, the round 2 starts.
    Round 2: the manager splits nodes into groups and each group contains
        two nodes, like [{0:8, 1:8},{2:8, 3:8}]. The node in each group will
        execute allgather independently and report its result to the manager.
        For example, the result is {0:False, 1:False, 2:True, 3:True}.
    Round 3: the manager will group the abnormal node with a normal node like
        [{0:8, 2:8}, {1:8, 2:8}]. Then, the node executes allgather again.
        If the result is {0:True, 1:False, 2:False, 3:True}, the network of
        node-1 if not available.
    """

    def __init__(self):
        super().__init__()
        self._name = RendezvousName.NETWORK_CHECK
        self._node_status: Dict[int, bool] = {}
        self._reported_nodes = set()
        self._node_groups: List[Dict[int, int]] = []

    def get_comm_world(self, rank_id):
        """Return the communication world if a round rendezvous is completed.
        The rendezvous is completed if one of the following conditions.
        """
        with self._lock:
            if not self._node_groups:
                rdzv_completed = self._check_rdzv_completed()
                if rdzv_completed:
                    self._node_groups = self._group_nodes(self._rdzv_round)
                    logger.info(
                        f"Round {self._rdzv_round} "
                        f"node group: {self._node_groups}"
                    )
                    if self._rdzv_round % 3 == 0:
                        self._node_status = {}
                    self._reported_nodes = set()
                    self._rdzv_round += 1

            for i, group in enumerate(self._node_groups):
                if rank_id in group:
                    return i, group
            return 0, {}

    def _group_nodes(self, round):
        """Group nodes into goups.
        Round 0: group all nodes into a group like {0:8, 1:8, 2:8, 3:8}.
        Round 1: Split nodes into groups and each group contains
            two nodes, like [{0:8, 1:8},{2:8, 3:8}].
        Round 1: group the abnormal node with a normal node like
            [{0:8, 2:8}, {1:8, 2:8}].
        """
        round = round % 3
        node_groups = []
        if round == 0:
            node_groups.append(self._rdzv_nodes)
        elif round == 1:
            group = {}
            for node_id, local_world_size in self._rdzv_nodes.items():
                group[node_id] = local_world_size
                if len(group) == 2:
                    node_groups.append(group)
                    group = {}
        elif round == 2:
            abnormal_nodes = []
            normal_nodes = []
            for node_id, status in self._node_status.items():
                if status:
                    normal_nodes.append(node_id)
                else:
                    abnormal_nodes.append(node_id)
            logger.info(
                f"Normal nodes: {normal_nodes}.\n"
                f"Abnormal nodes: {abnormal_nodes}"
            )
            if len(abnormal_nodes) > len(normal_nodes):
                return node_groups
            for i, node_id in enumerate(abnormal_nodes):
                group = {}
                group[node_id] = self._rdzv_nodes[node_id]
                group[normal_nodes[i]] = self._rdzv_nodes[node_id]
                node_groups.append(group)
            group = {}
            for node_id in normal_nodes[len(abnormal_nodes) :]:  # noqa: E203
                group[node_id] = self._rdzv_nodes[node_id]
            if group:
                node_groups.append(group)
        return node_groups

    def report_network_check_result(self, node_id: int, succeed):
        self._reported_nodes.add(node_id)
        self._node_status.setdefault(node_id, False)
        self._node_status[node_id] = self._node_status[node_id] or succeed
        if len(self._reported_nodes) == len(self._rdzv_nodes):
            logger.info(
                f"The {self._rdzv_round} network status of node "
                f"group is {self._node_status}."
            )

    def join_rendezvous(
        self,
        rank_id,
        local_world_size,
    ):
        """The node joins the current rond rendezvous.
        Args:
            rank_id: the node ID which is unique in an ElasticJob of DLrover.
            local_world_size: the local world size of a node.

        Returns:
            int: the number of rendezvous round.
        """
        self._node_groups = []
        return super().join_rendezvous(rank_id, local_world_size)

    def network_check_success(self):
        """Check the network task is succeed. Each task contains 3 rounds
        allgather. If succeed, the round should be set to the multiples of 3.
        """
        with self._lock:
            reason = ""
            success = False
            if len(self._reported_nodes) < len(self._rdzv_nodes):
                reason = NetworkFailureReason.WAITING_NODE
            else:
                success = self._node_status and all(
                    list(self._node_status.values())
                )
                if success:
                    self._rdzv_round = math.ceil(self._rdzv_round / 3) * 3
                else:
                    reason = NetworkFailureReason.NODE_FAILURE
            return success, reason
