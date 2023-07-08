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

from dlrover.python.common.constants import NetworkFailureReason
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node


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

    def update_rdzv_params(self, min_nodes, max_ndoes, waiting_timeout):
        """Update rendezvous parameters"""

    @abstractmethod
    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        pass

    @abstractmethod
    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        pass

    @abstractmethod
    def get_comm_world(self, node_id):
        """Get communication world of all alive nodes."""
        pass

    @abstractmethod
    def join_rendezvous(self, node_id, local_world_size):
        """The node joins a rond rendezvous."""
        pass

    @abstractmethod
    def report_network_check_result(self, node_id: int, normal: bool):
        """The node updates its status"""
        pass

    @abstractmethod
    def num_nodes_waiting(self):
        """Get the number of waiting nodes."""
        pass


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

    def update_rdzv_params(self, min_nodes, max_ndoes, waiting_timeout):
        """Update rendezvous parameters"""
        self._rdzv_params.min_nodes = min_nodes
        self._rdzv_params.max_nodes = max_ndoes
        self._rdzv_params.waiting_timeout = waiting_timeout

    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        self._alive_nodes.add(node.id)
        logger.info(
            f"Add alive worker {node.name} to elastic training rendezvous."
        )

    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        if node.id in self._alive_nodes:
            self._alive_nodes.remove(node.id)
            logger.info(f"Remove exited worker {node.name} from Rendezvous.")

    def get_released_workers(self):
        return []

    def get_comm_world(self, node_id):
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
            rdzv_completed = False
            if self._rdzv_nodes:
                return 0, self._rdzv_nodes
            if len(self._waiting_nodes) == self._rdzv_params.max_nodes:
                rdzv_completed = True
            else:
                waiting_num = len(self._waiting_nodes)
                alive_num = len(self._alive_nodes)
                waiting_time = time.time() - self._lastcall_time
                rdzv_completed = (
                    waiting_num >= self._rdzv_params.min_nodes
                    and waiting_num == alive_num
                    and waiting_time >= self._rdzv_params.waiting_timeout
                )

            if rdzv_completed:
                self._rdzv_nodes = dict(sorted(self._waiting_nodes.items()))
                self._waiting_nodes = dict()
                self._lastcall_time = 0
                logger.info(
                    f"Completed {self._rdzv_round} round "
                    f"rendezvous of elastic training is {self._rdzv_nodes}"
                )
                self._rdzv_round += 1

            return 0, self._rdzv_nodes

    def join_rendezvous(self, node_id, local_world_size):
        """The node joins the current rond rendezvous.
        Args:
            node_id: the node ID which is unique in an ElasticJob of DLrover.
            local_world_size: the local world size of a node.

        Returns:
            int: the number of rendezvous round.
        """
        with self._lock:
            if node_id in self._waiting_nodes:
                return
            self._waiting_nodes[node_id] = local_world_size
            self._rdzv_nodes = {}
            if len(self._waiting_nodes) >= self._rdzv_params.min_nodes:
                if self._lastcall_time == 0:
                    self._lastcall_time = time.time()
        return self._rdzv_round

    def num_nodes_waiting(self):
        """The number of waiting nodes. The agent of a node will re-join
        a rendezvous if it finds there are waiting nodes.
        """
        with self._lock:
            return len(self._waiting_nodes)

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
        self._node_status: Dict[int, bool] = {}
        self._reported_nodes = set()
        self._node_groups: List[Dict[int, int]] = []

    def update_rdzv_params(self, min_nodes, max_ndoes, waiting_timeout):
        """Update rendezvous parameters"""
        self._rdzv_params.min_nodes = min_nodes
        self._rdzv_params.max_nodes = max_ndoes
        self._rdzv_params.waiting_timeout = waiting_timeout

    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        self._alive_nodes.add(node.id)
        logger.info(
            f"Add alive worker {node.name} to network check rendezvous."
        )

    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        if node.id in self._alive_nodes:
            self._alive_nodes.remove(node.id)
            logger.info(f"Remove exited worker {node.name} from Rendezvous.")

    def get_released_workers(self):
        return []

    def get_comm_world(self, node_id):
        """Return the communication world if a round rendezvous is completed.
        The rendezvous is completed if one of the following conditions.
        """
        with self._lock:
            rdzv_completed = False
            if not self._node_groups:
                if len(self._waiting_nodes) == self._rdzv_params.max_nodes:
                    rdzv_completed = True
                else:
                    waiting_num = len(self._waiting_nodes)
                    alive_num = len(self._alive_nodes)
                    waiting_time = time.time() - self._lastcall_time
                    rdzv_completed = (
                        waiting_num >= self._rdzv_params.min_nodes
                        and waiting_num == alive_num
                        and waiting_time >= self._rdzv_params.waiting_timeout
                    )

                if rdzv_completed:
                    self._rdzv_nodes = dict(
                        sorted(self._waiting_nodes.items())
                    )
                    self._waiting_nodes = dict()
                    self._lastcall_time = 0
                    logger.info(
                        f"Completed {self._rdzv_round} round "
                        f"rendezvous of network check is {self._rdzv_nodes}"
                    )
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
                if node_id in group:
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
        node_id,
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
            if node_id in self._waiting_nodes:
                return
            self._waiting_nodes[node_id] = local_world_size
            self._rdzv_nodes = {}
            self._node_groups = []
            if len(self._waiting_nodes) >= self._rdzv_params.min_nodes:
                if self._lastcall_time == 0:
                    self._lastcall_time = time.time()
        return self._rdzv_round

    def num_nodes_waiting(self):
        with self._lock:
            return len(self._waiting_nodes)

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
