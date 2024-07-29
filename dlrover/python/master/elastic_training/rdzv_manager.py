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
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Tuple

from dlrover.python.common.constants import (
    NetworkFailureReason,
    RendezvousName,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.elastic_training.net_topology import (
    DefaultTopologyQuerier,
    DpTopologySorter,
    NodeTopologyMeta,
)


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
        # for both '_waiting_nodes' and '_rdzv_nodes', key is the node rank.
        self._waiting_nodes: Dict[int, NodeTopologyMeta] = {}
        self._rdzv_nodes: Dict[int, NodeTopologyMeta] = OrderedDict()
        self._lastcall_time = 0
        self._rdzv_params = RendezvousParameters(0, 0)
        self._rdzv_round = 0
        self._node_unit = 1
        self._name = ""
        self._latest_rdzv_nodes = []
        self._start_rdzv_ts = 0
        # key is the node rank, value is the time.
        self._node_rdzv_times: Dict[int, int] = {}
        self._latest_log_nodes_time = 0
        # key is the node rank, value is the step.
        self._save_ckpt_nodes: Dict[int, int] = {}
        self._topology_querier = DefaultTopologyQuerier()
        self._topology_sorter = DpTopologySorter()

    def get_min_nodes(self):
        return self._rdzv_params.min_nodes

    def get_rdzv_round(self):
        return self._rdzv_round

    def clear_waiting_nodes(self):
        self._waiting_nodes.clear()

    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        self._alive_nodes.add(node.id)

    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        if node.id in self._alive_nodes:
            self._alive_nodes.remove(node.id)
            self._has_node_failed = True
        remove_rank = -1
        for rank, meta in self._waiting_nodes.items():
            if meta.node_id == node.id:
                remove_rank = rank
                break
        if remove_rank > 0:
            self._waiting_nodes.pop(remove_rank, None)
            logger.info(
                f"Remove exited worker {node.name} with rank {remove_rank} "
                f" from {self._name} rendezvous."
            )

    def update_rdzv_params(
        self, min_nodes, max_ndoes, waiting_timeout, node_unit
    ):
        """Update rendezvous parameters
        Args:
            min_nodes: The minimum number of nodes.
            max_nodes: THe maximum number of nodes.
            waiting_timeout: the time to wait more workers.
            node_unit: the number unit of workers to build the communication
                world. This is, the number of nodes in a world should be
                a multiple of worker_unit.
        """
        with self._lock:
            if self._rdzv_params.max_nodes == 0:
                self._rdzv_params.min_nodes = min_nodes
                self._rdzv_params.max_nodes = max_ndoes
                self._rdzv_params.waiting_timeout = waiting_timeout
                self._node_unit = node_unit
                logger.info(
                    f"{self._name} manager updates rdzv params: "
                    f"min_nodes={min_nodes}, max_nodes={max_ndoes}, "
                    f"waiting_timeout={waiting_timeout}, node_unit={node_unit}"
                )

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
            self._rdzv_nodes = OrderedDict()
            for i in node_ids:
                self._rdzv_nodes[i] = self._waiting_nodes[i]
            self._latest_rdzv_nodes = list(self._rdzv_nodes.keys())
            extra_nodes = {}
            for i in self._waiting_nodes.keys():
                if i not in self._rdzv_nodes:
                    extra_nodes[i] = self._waiting_nodes[i]
            self._waiting_nodes = extra_nodes
            self._lastcall_time = 0
            self._log_rendezvous_info()
            if self._waiting_nodes:
                waiting_node_ids = []
                for node in self._waiting_nodes.values():
                    waiting_node_ids.append(node.node_id)
                logger.warning(
                    f"Waiting nodes not in {self._rdzv_round} rendezvous "
                    f"are {waiting_node_ids}."
                )
        elif time.time() - self._latest_log_nodes_time > 60:
            self._latest_log_nodes_time = time.time()
            waiting_nodes = {}
            for rank, node in self._waiting_nodes.items():
                waiting_nodes[node.node_id] = rank
            lacking_ranks = self._get_lacking_ranks()
            logger.info(f"Waiting nodes(required:{self._rdzv_params.min_nodes}"
                        f"/{self._rdzv_params.max_nodes}) in rendezvous(size:"
                        f"{len(waiting_nodes)}) are {waiting_nodes}, "
                        f"lacking ranks(size:{len(lacking_ranks)}) "
                        f"are {lacking_ranks}")
        return rdzv_completed

    def _get_lacking_ranks(self):
        """
        Lacking ranks = min required nodes - waiting ranks.

        Return:
            ranks number in list
        """

        lacking_ranks = []
        if self._rdzv_params is None or self._rdzv_params.min_nodes <= 0:
            return lacking_ranks

        min_required = self._rdzv_params.min_nodes
        min_ranks = set([i for i in range(min_required)])
        if self._waiting_nodes:
            waiting_ranks = set(self._waiting_nodes.keys())
        else:
            waiting_ranks = set([])

        if len(min_ranks) > len(waiting_ranks):
            lacking_ranks = list(min_ranks - waiting_ranks)

        return lacking_ranks

    def _log_rendezvous_info(self):
        node_ranks = {}
        for rank, node in self._rdzv_nodes.items():
            node_ranks[node.node_id] = rank
        node_ranks = dict(sorted(node_ranks.items()))
        node_rdzv_times = self._map_node_rank_to_id(self._node_rdzv_times)
        logger.info(
            f"Completed {self._rdzv_round} round "
            f"rendezvous of {self._name} is {node_ranks} \n"
            "The times of nodes to join rendezvous "
            f"are {node_rdzv_times}."
        )
        self._node_rdzv_times.clear()
        if self._start_rdzv_ts > 0:
            rdzv_time = round(time.time() - self._start_rdzv_ts, 2)
            logger.info(
                f"Elapsed time to complete the {self._rdzv_round} "
                f"round rendzvous is {rdzv_time}s"
            )
        self._start_rdzv_ts = 0

    def not_joined_rdzv_nodes(self):
        """Return workers which do not join a rendezvous."""
        nodes = []
        join_node_ids = []
        for node in self._rdzv_nodes.values():
            join_node_ids.append(node.node_id)
        if self._rdzv_nodes:
            for node_id in self._alive_nodes:
                if node_id not in join_node_ids:
                    nodes.append(node_id)
        return nodes

    def join_rendezvous(
        self,
        node_id,
        node_rank,
        local_world_size,
        node_ip="",
    ):
        """The node joins the current rond rendezvous.
        Args:
            node_id: the node ID which is unique in an ElasticJob of DLrover.
            local_world_size: the local world size of a node.

        Returns:
            int: the number of rendezvous round.
        """
        with self._lock:
            if not self._waiting_nodes:
                self._start_rdzv_ts = time.time()
            if node_rank in self._waiting_nodes:
                logger.info(
                    f"Skip rdzv joining for node : {node_rank} "
                    "because the target node is no longer in the "
                    "waiting nodes list."
                )
                return self._rdzv_round
            asw, psw = self._topology_querier.query(node_ip)
            meta = NodeTopologyMeta(
                node_id=node_id,
                node_rank=node_rank,
                node_ip=node_ip,
                process_num=local_world_size,
                asw=asw,
                psw=psw,
            )
            self._waiting_nodes[node_rank] = meta
            self._rdzv_nodes = OrderedDict()
            self._lastcall_time = time.time()
            self._node_rdzv_times[node_rank] = round(
                self._lastcall_time - self._start_rdzv_ts, 2
            )

        return self._rdzv_round

    def _map_node_rank_to_id(self, rank_dict):
        """
        Convert the dict with the node rank as the key to a dict
        with the node id. Because, it is more clear to show the node
        id in the log than the node rank. If the log shows the node rank,
        users need to search which node has the node rank.
        """
        id_dict = {}
        for node_rank, v in rank_dict.items():
            if node_rank not in self._rdzv_nodes:
                continue
            node_id = self._rdzv_nodes[node_rank].node_id
            id_dict[node_id] = v
        id_dict = dict(sorted(id_dict.items()))
        return id_dict

    def num_nodes_waiting(self):
        """The elastic agent will restart training processes if it
        find the number of waiting nodes is not zero. The manager
        will notify all nodes to restart training processes immediately if
        an existing node re-joins the next round rendezvous.
        If there are new nodes, the master notifies all nodes to re-join
        the next round rendezvous only when the number of waiting nodes
        is bigger than the number unit of nodes.
        """
        if self._has_node_restart():
            return len(self._waiting_nodes)
        elif len(self._waiting_nodes) >= self._node_unit:
            return len(self._waiting_nodes)
        return 0

    def _has_node_restart(self):
        """The node will restart training processes if it
        re-joins the rendezvous."""
        for node_rank in self._waiting_nodes.keys():
            if node_rank in self._latest_rdzv_nodes:
                return True
        return False

    def sync_ckpt_nodes(self, node_id, step):
        self._save_ckpt_nodes[node_id] = step
        steps = set(self._save_ckpt_nodes.values())
        if len(steps) > 1:
            return False
        if len(self._save_ckpt_nodes) == len(self._latest_rdzv_nodes):
            return True
        return False

    @abstractmethod
    def get_comm_world(
        self, node_rank
    ) -> Tuple[int, int, Dict[int, NodeTopologyMeta]]:
        """Get communication world of all alive nodes.

        Args:
            node_rank: the id of node.

        Returns:
            rdzv_round: the round index.
            group: the group index.
            world: Dict like {0: 8, 1: 8, 2: 8} where the key is the rank ID
            and the value is the local world size of the node.
        """
        pass

    @abstractmethod
    def report_network_check_result(
        self, node_rank: int, normal: bool, elapsed_time: float
    ):
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

    def get_comm_world(
        self, node_rank
    ) -> Tuple[int, int, Dict[int, NodeTopologyMeta]]:
        """Return the communication world if a round rendezvous is completed.
        The rendezvous is completed if one of the following conditions
        is satisfied:
        1. The size of waiting node list is equal to the max_nodes.
        2. The size of waiting node list is bigger than the min_nodes and
            equal to the size of alive node list. What's more, no more worker
            join the rendezvous in waiting_timeout.

        Returns:
            rdzv_round: the round index.
            group: the group index.
            world: Dict like {0: 8, 1: 8, 2: 8} where the key is the rank ID
            and the value is the local world size of the node.
        """
        with self._lock:
            if not self._rdzv_nodes:
                rdzv_completed = self._check_rdzv_completed()
                if rdzv_completed:
                    self._rdzv_round += 1
                    self._rdzv_nodes = self._topology_sorter.sort(
                        self._rdzv_nodes
                    )
                    ranks = list(self._rdzv_nodes.keys())
                    node_ips = []
                    node_ids = []
                    for node_rank in ranks:
                        node = self._rdzv_nodes[node_rank]
                        node_ips.append(node.node_ip)
                        node_ids.append(node.node_id)
                    logger.info(
                        f"Node ids are {node_ids}.\n Node IPs are {node_ips}"
                    )
            return self._rdzv_round, 0, self._rdzv_nodes

    def report_network_check_result(self, node_rank, normal, elapsed_time):
        return


class NetworkCheckRendezvousManager(RendezvousManager):
    """NcclCheckRendezvousManager runs on the DLRover master. The task
    to check network contains 2 round to execute allgather on all nodes.
    We show the detail to check network assuming there are 4 nodes.
    Round 0: the manager splits nodes into groups and each group contains
        two nodes, like [{0:8, 1:8},{2:8, 3:8}]. The node in each group will
        execute allgather independently and report its result to the manager.
        For example, the result is {0:False, 1:False, 2:True, 3:True}.
    Round 1: the manager will group the abnormal node with a normal node like
        [{0:8, 2:8}, {1:8, 2:8}]. Then, the node executes allgather again.
        If the result is {0:True, 1:False, 2:False, 3:True}, the network of
        node-1 if not available.
    """

    def __init__(self):
        super().__init__()
        self._name = RendezvousName.NETWORK_CHECK
        self._node_status: Dict[int, bool] = {}
        self._node_times: Dict[int, float] = {}
        self._reported_nodes = set()
        self._node_groups: List[Dict[int, NodeTopologyMeta]] = []
        self._check_round = 2
        self._fault_nodes = set()
        self._straggler_nodes = set()

    def _get_print_node_groups(self):
        printing_node_groups = []
        for group in self._node_groups:
            ids = [
                self._rdzv_nodes[rank].node_id
                for rank in group.keys()
            ]
            printing_node_groups.append(ids)

        return printing_node_groups

    def get_comm_world(
        self, node_rank
    ) -> Tuple[int, int, Dict[int, NodeTopologyMeta]]:
        """Return the communication world if a round rendezvous is completed.
        The rendezvous is completed if one of the following conditions.
        """
        with self._lock:
            if not self._node_groups:
                rdzv_completed = self._check_rdzv_completed()
                if rdzv_completed:
                    self._fault_nodes.clear()
                    self._straggler_nodes.clear()
                    self._node_groups = self._group_nodes(self._rdzv_round)
                    logger.info(
                        f"Node groups of round {self._rdzv_round} "
                        f"are: {self._get_print_node_groups()}."
                    )
                    if self._rdzv_round % 2 == 0:
                        self._clear_check_status()
                    self._reported_nodes = set()
                    self._rdzv_round += 1
            for i, group in enumerate(self._node_groups):
                if node_rank in group:
                    return self._rdzv_round, i, group
            return self._rdzv_round, 0, self._rdzv_nodes

    def _clear_check_status(self):
        self._node_status = {}
        self._node_times = {}

    def _group_nodes(self, round):
        """Group nodes into goups.
        Round 0: group all nodes into a group like {0:8, 1:8, 2:8, 3:8}.
        Round 1: Split nodes into groups and each group contains
            two nodes, like [{0:8, 1:8},{2:8, 3:8}].
        Round 1: group the abnormal node with a normal node like
            [{0:8, 2:8}, {1:8, 2:8}].
        """
        round = round % self._check_round
        node_groups: List[Dict[int, int]] = []
        if round == 0:
            group = {}
            for node_id, meta in self._rdzv_nodes.items():
                group[node_id] = meta
                if len(group) == 2:
                    node_groups.append(group)
                    group = {}
            if len(group) == 1:
                if len(node_groups) > 0:
                    node_groups[-1].update(group)
                else:
                    node_groups.append(group)
        elif round == 1:
            self._check_abnormal_nodes()
            node_times = sorted(self._node_times.items(), key=lambda x: x[1])
            cur_nodes = []
            for node_id, _ in node_times:
                if node_id in self._rdzv_nodes:
                    cur_nodes.append(node_id)
            left, right = 0, len(cur_nodes) - 1
            group = {}
            while right >= left:
                group = {}
                node0 = cur_nodes[left]
                node1 = cur_nodes[right]
                group[node0] = self._rdzv_nodes[node0]
                group[node1] = self._rdzv_nodes[node1]
                if len(group) == 2:
                    node_groups.append(group)
                left += 1
                right -= 1
            if len(group) == 1:
                if len(node_groups) > 0:
                    node_groups[-1].update(group)
                else:
                    node_groups.append(group)
        return node_groups

    def _check_abnormal_nodes(self):
        abnormal_nodes = []
        normal_nodes = []
        for node_rank, status in self._node_status.items():
            node_id = self._rdzv_nodes[node_rank].node_id
            if status:
                normal_nodes.append(node_id)
            else:
                abnormal_nodes.append(node_id)
        logger.info(
            f"Normal nodes: {normal_nodes}.\n"
            f"Abnormal nodes: {abnormal_nodes}"
        )

    def report_network_check_result(
        self, node_rank: int, succeed: bool, elapsed_time: float
    ):
        self._reported_nodes.add(node_rank)
        self._node_status.setdefault(node_rank, succeed)
        self._node_times.setdefault(node_rank, elapsed_time)
        self._node_status[node_rank] = self._node_status[node_rank] or succeed
        self._node_times[node_rank] = round(
            min(self._node_times[node_rank], elapsed_time), 3
        )
        if len(self._reported_nodes) == len(self._rdzv_nodes):
            node_status = self._map_node_rank_to_id(self._node_status)
            logger.info(
                f"Round {self._rdzv_round}: The node status "
                f"are: {node_status}, "
                f"the node group are: {self._get_print_node_groups()}"
            )
            node_check_times = self._map_node_rank_to_id(self._node_times)
            logger.info(
                f"Round {self._rdzv_round}: The node elapsed time "
                f"are {node_check_times}"
            )

    def join_rendezvous(
        self,
        node_id,
        node_rank,
        local_world_size,
        node_ip="",
    ):
        """The node joins the current rond rendezvous.
        Args:
            node_rank: the node rank which is unique in an
                ElasticJob of DLrover.
            local_world_size: the local world size of a node.

        Returns:
            int: the number of rendezvous round.
        """
        self._node_groups.clear()
        return super().join_rendezvous(
            node_id, node_rank, local_world_size, node_ip
        )

    def check_fault_node(self):
        """Check whether the job has fault nodes. Each task contains 2 rounds
        allgather. If succeed, the round should be set to the multiples of 2.
        """
        with self._lock:
            if not self._rdzv_nodes:
                logger.warning(
                    "Skip check for rdzv_nodes hasn't been initialized."
                )
            reason = ""
            all_joined = len(self._reported_nodes) >= len(self._rdzv_nodes)
            if not all_joined:
                reason = NetworkFailureReason.WAITING_NODE
            elif len(self._fault_nodes) == 0:
                for node_rank, status in self._node_status.items():
                    if not status:
                        self._fault_nodes.add(node_rank)
                if len(self._fault_nodes) > 0:
                    fault_nodes = {}
                    for rank in self._fault_nodes:
                        fault_nodes[rank] = self._rdzv_nodes[rank].node_id
                    logger.warning("Fault nodes(rank:node_id) "
                                   f"are: {fault_nodes}")
                stragglers = self._detect_stragglers()
                if not self._fault_nodes and not stragglers:
                    self._rdzv_round = (
                        math.ceil(self._rdzv_round / self._check_round)
                        * self._check_round
                    )
            if all_joined and len(self._fault_nodes) > 0:
                reason = NetworkFailureReason.NODE_FAILURE
            return list(self._fault_nodes), reason

    def get_straggler(self):
        """Detect whether there is the straggler according to the
        elapsed time of node to run the test task. If the elapsed
        time of node is bigger than 2*median_time, the node is
        a straggler.
        """
        with self._lock:
            reason = ""
            stragglers: Dict[int, float] = {}
            if len(self._reported_nodes) < len(self._rdzv_nodes):
                reason = NetworkFailureReason.WAITING_NODE
            elif len(self._straggler_nodes) == 0:
                stragglers = self._detect_stragglers()
                if stragglers:
                    logger.warning(f"Straggler: {stragglers}.")
                self._straggler_nodes.update(stragglers)
            return list(self._straggler_nodes), reason

    def _detect_stragglers(self):
        """Detect wether there is the straggler in the job."""
        stragglers: Dict[int, float] = {}
        times = sorted(list(self._node_times.values()))
        if not times:
            return stragglers
        if len(times) % 2 == 0:
            i = len(times) // 2
            med_time = (times[i] + times[i - 1]) / 2
        else:
            i = len(times) // 2
            med_time = times[i]
        for node_id, t in self._node_times.items():
            if t > med_time * 2:
                stragglers[node_id] = t
        return stragglers
