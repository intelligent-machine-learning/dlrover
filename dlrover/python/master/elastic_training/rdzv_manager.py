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
import random
import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from threading import RLock
from typing import Dict, List, Optional, Tuple

from dlrover.python.common.constants import (
    EventReportConstants,
    JobConstant,
    NetworkFailureReason,
    RendezvousName,
    NodeType,
)
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.elastic_training.net_topology import (
    DefaultTopologyQuerier,
    DpTopologySorter,
    NodeTopologyMeta,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.training_event import DLRoverMasterEvent
from dlrover.python.training_event.emitter import DurationSpan

_master_evt = DLRoverMasterEvent().singleton_instance()
job_ctx = get_job_context()


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
        self._lock = RLock()
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
        self._event_reporter = get_event_reporter()
        self.rendezvous_events: Dict[int, DurationSpan] = {}
        self._rdzv_blocked = False
        self._rdzv_block_reason = ""
        self._rdzv_completed_callbacks = []

    def get_min_nodes(self):
        return self._rdzv_params.min_nodes

    def get_max_nodes(self):
        return self._rdzv_params.max_nodes

    def get_waiting_timeout(self):
        return self._rdzv_params.waiting_timeout

    def get_rdzv_round(self):
        return self._rdzv_round

    def set_rdzv_blocked(self, blocked: bool, reason: Optional[str] = None):
        with self._lock:
            self._rdzv_blocked = blocked
            if blocked:
                self._rdzv_block_reason = reason or ""
            else:
                self._rdzv_block_reason = ""

    def is_rdzv_blocked(self) -> Tuple[bool, Optional[str]]:
        with self._lock:
            if not self._rdzv_blocked:
                return False, None
            return True, self._rdzv_block_reason or None

    def _pre_rdzv_check_hook(self) -> Tuple[bool, Optional[str]]:
        """Hook to block rendezvous completion.

        Returns:
            (blocked, reason). Subclasses can override to add custom checks.
        """
        return self.is_rdzv_blocked()

    def clear_waiting_nodes(self):
        with self._lock:
            self._waiting_nodes.clear()

    def add_alive_node(self, node: Node):
        """When a node is running, the master will add it to alive list."""
        self._alive_nodes.add(node.id)

    def remove_alive_node(self, node: Node):
        """When a node is exited, the master will remove it from alive list."""
        if node.id in self._alive_nodes:
            self._alive_nodes.remove(node.id)

        with self._lock:
            remove_rank = -1
            for rank, meta in self._waiting_nodes.items():
                if meta.node_id == node.id:
                    remove_rank = rank
                    break
            if remove_rank >= 0:
                removed = self._waiting_nodes.pop(remove_rank, None)
                if removed is not None:
                    logger.info(
                        f"Remove exited worker {removed.node_id} "
                        f"with rank {remove_rank} "
                        f"from {self._name} rendezvous."
                    )

    def update_rdzv_params(
        self, min_nodes, max_nodes, waiting_timeout, node_unit
    ):
        """Update rendezvous parameters
        Args:
            min_nodes: The minimum number of nodes.
            max_nodes: The maximum number of nodes.
            waiting_timeout: the time to wait more workers.
            node_unit: the number unit of workers to build the communication
                world. This is, the number of nodes in a world should be
                a multiple of worker_unit.
        """
        with self._lock:
            if self._rdzv_params.max_nodes == 0:
                self._rdzv_params.min_nodes = min_nodes
                self._rdzv_params.max_nodes = max_nodes
                self._rdzv_params.waiting_timeout = waiting_timeout
                self._node_unit = node_unit
                logger.info(
                    f"{self._name} manager updates rdzv params: "
                    f"min_nodes={min_nodes}, max_nodes={max_nodes}, "
                    f"waiting_timeout={waiting_timeout}, node_unit={node_unit}"
                )

    def _check_rdzv_completed(self):
        # check node aliveness according to reported status because the actual
        # pod status might be delayed after node exiting
        waiting_nodes_id = [
            node_topo.node_id
            for node_topo in list(self._waiting_nodes.values())
        ]
        for waiting_node_id in waiting_nodes_id:
            target_node = job_ctx.job_node(NodeType.WORKER, waiting_node_id)
            if target_node and target_node.is_failed_and_exited():
                self.remove_alive_node(target_node)

        rdzv_completed = False
        # if the universal checkpoint before scaling out is not ready,
        # the next round rendezvous cannot be completed
        blocked, reason = self._pre_rdzv_check_hook()
        if blocked:
            if reason:
                logger.info(reason)
            return False
        waiting_num = len(self._waiting_nodes)
        if waiting_num == self._rdzv_params.max_nodes:
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
            logger.info(
                f"Waiting nodes(required:{self._rdzv_params.min_nodes}"
                f"/{self._rdzv_params.max_nodes}) in rendezvous(size:"
                f"{len(waiting_nodes)}) are {waiting_nodes}, "
                f"lacking ranks(size:{len(lacking_ranks)}) "
                f"are {lacking_ranks} for round {self._rdzv_round}"
            )
        return rdzv_completed

    def _get_lacking_ranks(self) -> List[int]:
        """
        Lacking ranks = min required nodes(ranks) - waiting ranks.

        Return:
            ranks in list e.g.[5, 6]
        """

        lacking_ranks: List[int] = []
        if (
            self._rdzv_params is None
            or self._rdzv_params.min_nodes <= 0
            or self._rdzv_params.max_nodes <= 0
        ):
            return lacking_ranks

        max_required = self._rdzv_params.max_nodes
        min_ranks = set([i for i in range(max_required)])
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
            logger.info(
                f"Worker node with id: {meta.node_id}, "
                f"rank: {meta.node_rank} and ip: {meta.node_ip} "
                f"joining {self._name} rendezvous for round: {self._rdzv_round}."
            )
            self._waiting_nodes[node_rank] = meta
            self._rdzv_nodes = OrderedDict()
            self._lastcall_time = time.time()
            self._node_rdzv_times[node_rank] = round(
                self._lastcall_time - self._start_rdzv_ts, 2
            )
            if self._rdzv_round not in self.rendezvous_events.keys():
                self.rendezvous_events[self._rdzv_round] = self.new_rdzv_evt()
            self._event_reporter.report_rdzv_node_join(
                meta,
                self.rendezvous_events[self._rdzv_round],
                self._name,
                self._rdzv_round,
                self._rdzv_params,
                waiting_nodes=self._waiting_nodes,
                node_elapsed_time=self._node_rdzv_times[node_rank],
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

    def new_rdzv_evt(self):
        return _master_evt.rendezvous(
            rendezvous_type=self._name,
            round_num=self.get_rdzv_round(),
            timeout_sec=self.get_waiting_timeout(),
            max_nodes=self.get_max_nodes(),
            min_nodes=self.get_min_nodes(),
        )

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

    def add_rdzv_completed_callback(self, callback):
        """
        Callback when rdzv completed.

        Callback signature: (rdzv_round: int, rdzv_nodes: List[int)
        """
        self._rdzv_completed_callbacks.append(callback)

    def process_error(
        self, node_id, node_rank, err_type, err_message, elapsed_time
    ):
        if self._rdzv_round in self.rendezvous_events.keys():
            self._event_reporter.report_rdzv_timeout(
                self.rendezvous_events[self._rdzv_round],
                self._name,
                self._rdzv_round,
                self._rdzv_params,
                node_id=node_id,
                node_rank=node_rank,
                node_groups=[],
                elapsed_time=elapsed_time,
                error_type=err_type,
                error_message=err_message,
            )


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
        self._name = RendezvousName.TRAINING

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
                    finished_rdzv_round = self._rdzv_round
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
                    node_elapsed_time = time.time() - self._lastcall_time

                    if finished_rdzv_round in self.rendezvous_events.keys():
                        self._event_reporter.report_rdzv_complete(
                            self.rendezvous_events[finished_rdzv_round],
                            self._name,
                            finished_rdzv_round,
                            self._rdzv_params,
                            node_ids=node_ids,
                            node_rank=node_rank,
                            node_elapsed_time=node_elapsed_time,
                        )
                    self._on_rdzv_completed(
                        finished_rdzv_round,
                        node_ids,
                    )

            return self._rdzv_round, 0, self._rdzv_nodes

    def report_network_check_result(self, node_rank, normal, elapsed_time):
        return

    def _on_rdzv_completed(self, rdzv_round, rdzv_nodes):
        for callback in self._rdzv_completed_callbacks:
            try:
                callback(rdzv_round, rdzv_nodes)
            except Exception as e:
                logger.warning(f"Rendezvous completed callback error: {e}")


class UcpRdzvManager(ElasticTrainingRendezvousManager):
    """UcpRdzvManager blocks rendezvous completion until previous round ends."""

    def __init__(self):
        super().__init__()

    def set_rdzv_blocked(self, blocked: bool, reason: Optional[str] = None):
        if blocked and not reason:
            reason = (
                f"Previous rendezvous round ({self._rdzv_round}) not finished yet. "
                f"blocked={blocked}. "
                f"Waiting for previous round completion."
            )
        super().set_rdzv_blocked(blocked, reason)


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

    # Round-0 (coverage) pairing modes, resolved by _pre_rdzv_check_hook and
    # consumed by _group_round0.
    _PAIRING_NO_GROUP = "no_group"
    _PAIRING_WITH_GROUP = "with_group"

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
        self._network_check_evt = self.new_network_check_evt()
        # Group-aware pairing state. The scheduler patches the
        # scheduling/rack-id label after a pod is created, so at rdzv
        # completion some workers may already carry group info and some may
        # not. _pre_rdzv_check_hook gates completion until the group info is
        # consistent (all-have / all-none) or the sync timeout elapses; the
        # resolved mode is then consumed by _group_round0.
        self._group_pairing_mode: Optional[str] = None
        self._group_eval_ts: float = 0.0
        self._group_eval_blocked: bool = False
        self._group_eval_reason: str = ""
        self._partial_start_ts: float = 0.0

    def new_network_check_evt(self):
        return _master_evt.network_check(round=self.get_rdzv_round())

    def _get_print_node_groups(self):
        printing_node_groups = []
        for group in self._node_groups:
            ids = [self._rdzv_nodes[rank].node_id for rank in group.keys()]
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
                    print_node_groups = self._get_print_node_groups()
                    logger.info(
                        f"Node groups of round {self._rdzv_round} "
                        f"are: {print_node_groups}."
                    )
                    if self._rdzv_round % 2 == 0:
                        self._clear_check_status()
                    self._reported_nodes = set()
                    self._network_check_evt = self.new_network_check_evt()
                    self._network_check_evt.begin()

                    finished_rdzv_round = self._rdzv_round
                    self._rdzv_round += 1
                    elapsed_time = time.time() - self._lastcall_time

                    if finished_rdzv_round in self.rendezvous_events.keys():
                        self._event_reporter.report_rdzv_complete(
                            self.rendezvous_events[finished_rdzv_round],
                            self._name,
                            finished_rdzv_round,
                            self._rdzv_params,
                            node_ids=print_node_groups,
                            node_rank=node_rank,
                            elapsed_time=elapsed_time,
                        )

            for i, group in enumerate(self._node_groups):
                if node_rank in group:
                    return self._rdzv_round, i, group
            return self._rdzv_round, 0, self._rdzv_nodes

    def _clear_check_status(self):
        self._node_status = {}
        self._node_times = {}

    def _query_node_group(self, node_id):
        """Query the group info of a node from the job context.

        Returns:
            Tuple (group_index, group_size, group_id). Returns (-1, 0, "")
            when the node is unknown or has no group info yet, e.g. the
            scheduler has not yet patched the scheduling/rack-id label.
        """
        node = job_ctx.job_node(NodeType.WORKER, node_id)
        if node and node.has_group():
            return node.group, node.group_size, node.group_id
        return -1, 0, ""

    def _count_group_consistency(self):
        """Count joined workers that have / lack group info right now.

        Returns (have_count, none_count, none_node_ids). Reads the live
        job_ctx which the master's periodic pod LIST keeps fresh, so
        repeated calls observe scheduler-patched labels as they arrive.
        """
        have = 0
        none_node_ids: List[int] = []
        for meta in self._waiting_nodes.values():
            group_idx, _, _ = self._query_node_group(meta.node_id)
            if group_idx >= 0:
                have += 1
            else:
                none_node_ids.append(meta.node_id)
        return have, len(none_node_ids), none_node_ids

    def _reset_group_gate_state(self):
        """Clear partial-wait tracking so the next round-0 re-evaluates."""
        self._partial_start_ts = 0.0
        self._group_eval_ts = 0.0
        self._group_eval_blocked = False
        self._group_eval_reason = ""

    def _pre_rdzv_check_hook(self) -> Tuple[bool, Optional[str]]:
        """Gate rdzv completion on group-info consistency for round 0.

        The scheduler patches the scheduling/rack-id label after pod
        creation, so at completion some joined workers may already carry
        group info while others do not. Round-0 pairing needs the group info
        to be consistent: all-have -> balanced intra/inter pairing; all-none
        -> legacy consecutive pairing. This hook blocks completion (without
        sleeping, since it runs under self._lock) until consistency is
        reached or NETWORK_CHECK_GROUP_SYNC_TIMEOUT elapses, after which it
        falls back to no-group. Retries are driven by agents polling
        get_comm_world; the master's periodic pod LIST refreshes job_ctx.
        """
        # Only the round-0 (coverage) pairing relies on group info; the
        # round-1 diagnostic pairing is group-agnostic.
        if self._rdzv_round % self._check_round != 0:
            return self.is_rdzv_blocked()
        # Gate only once all expected workers have joined.
        if len(self._waiting_nodes) < self._rdzv_params.max_nodes:
            return self.is_rdzv_blocked()

        now = time.time()
        # Throttle re-evaluation: the lock is held and every agent polls.
        interval = JobConstant.NETWORK_CHECK_GROUP_SYNC_INTERVAL
        if now - self._group_eval_ts < interval:
            return self._group_eval_blocked, (self._group_eval_reason or None)
        self._group_eval_ts = now

        have, none, none_ids = self._count_group_consistency()
        if none == 0:
            logger.info(
                f"network-check group info ready: all {have} joined workers "
                f"carry group info; use balanced intra/inter pairing."
            )
            self._group_pairing_mode = self._PAIRING_WITH_GROUP
            self._reset_group_gate_state()
            self._group_eval_blocked = False
            self._group_eval_reason = ""
            return False, None
        if have == 0:
            logger.info(
                "network-check: no joined worker carries group info "
                "(scheduler did not patch labels); use consecutive pairing."
            )
            self._group_pairing_mode = self._PAIRING_NO_GROUP
            self._reset_group_gate_state()
            self._group_eval_blocked = False
            self._group_eval_reason = ""
            return False, None

        # Partial: some have, some lack group info. Wait for the watcher to
        # sync the remaining scheduler-patched labels.
        timeout = JobConstant.NETWORK_CHECK_GROUP_SYNC_TIMEOUT
        if self._partial_start_ts == 0.0:
            self._partial_start_ts = now
            logger.info(
                f"network-check group info partial: have={have} none={none} "
                f"(missing node_ids sample={none_ids[:16]}); wait up to "
                f"{timeout}s for scheduler-patched rack-id labels to sync."
            )
        elapsed = now - self._partial_start_ts
        if elapsed >= timeout:
            logger.warning(
                f"network-check group info still partial after {timeout}s "
                f"(have={have} none={none}, missing node_ids sample="
                f"{none_ids[:16]}); fall back to consecutive pairing."
            )
            self._group_pairing_mode = self._PAIRING_NO_GROUP
            self._reset_group_gate_state()
            self._group_eval_blocked = False
            self._group_eval_reason = ""
            return False, None
        self._group_eval_blocked = True
        self._group_eval_reason = "waiting for network-check group info sync"
        logger.info(
            f"network-check waiting group info sync: have={have} none={none} "
            f"elapsed={int(elapsed)}s"
        )
        return True, self._group_eval_reason

    def _group_nodes(self, round):
        """Group nodes into pairs for the network check.

        Round 0 (coverage): pair up all nodes. When group info is available
        and consistent, each group contributes half of its nodes to
        intra-group pairs (ranks shuffled) and half to inter-group pairs
        (balanced across groups), so both the intra-fabric (e.g. NVSwitch)
        and the inter-fabric (e.g. IB) get exercised in one round. Without
        group info, falls back to the legacy consecutive-rank pairing. The
        mode is resolved by _pre_rdzv_check_hook.
        Round 1 (diagnostic): pair each abnormal node with a normal node to
        locate the faulty node (group-agnostic).
        """
        round = round % self._check_round
        node_groups: List[Dict[int, int]] = []
        if round == 0:
            return self._group_round0()
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

    def _consecutive_pairs(
        self, ranks: List[int]
    ) -> List[Dict[int, NodeTopologyMeta]]:
        """Legacy round-0 pairing: consecutive ranks {r0,r1},{r2,r3},...

        A trailing odd node is folded into the last pair, matching the
        original behaviour.
        """
        node_groups: List[Dict[int, NodeTopologyMeta]] = []
        group: Dict[int, NodeTopologyMeta] = {}
        for rank in ranks:
            group[rank] = self._rdzv_nodes[rank]
            if len(group) == 2:
                node_groups.append(group)
                group = {}
        if len(group) == 1:
            if node_groups:
                node_groups[-1].update(group)
            else:
                node_groups.append(group)
        logger.info(
            f"network-check round {self._rdzv_round} coverage pairing "
            f"(mode=no_group): consecutive-rank pairs={len(node_groups)}, "
            f"nodes={len(ranks)}."
        )
        return node_groups

    def _group_round0(self) -> List[Dict[int, NodeTopologyMeta]]:
        """Build round-0 (coverage) pairs.

        With group info: each group's nodes are split in half; one half is
        paired within the group (ranks shuffled to break consecutive-rank
        bias) and the other half is paired across groups (balanced so each
        group tests against different groups on average). Each node joins
        exactly one pair, so a single round exercises both fabrics
        fleet-wide.

        Without group info (or after the consistency-gate timeout): legacy
        consecutive-rank pairing.
        """
        mode = self._group_pairing_mode or self._PAIRING_NO_GROUP
        ranks = list(self._rdzv_nodes.keys())  # ascending rank order
        if len(ranks) < 2:
            return self._consecutive_pairs(ranks)
        if mode != self._PAIRING_WITH_GROUP:
            return self._consecutive_pairs(ranks)

        # Build group -> [ranks], reading group info live from job_ctx.
        group_map: Dict[int, List[int]] = {}
        insertion_order: List[int] = []
        ungrouped: List[int] = []
        for rank in ranks:
            meta = self._rdzv_nodes[rank]
            group_idx, _, _ = self._query_node_group(meta.node_id)
            if group_idx < 0:
                ungrouped.append(rank)
            else:
                if group_idx not in group_map:
                    group_map[group_idx] = []
                    insertion_order.append(group_idx)
                group_map[group_idx].append(rank)

        if ungrouped:
            logger.warning(
                f"network-check round 0: {len(ungrouped)} node(s) still lack "
                f"group info at pairing time (ranks sample={ungrouped[:16]}); "
                f"they are folded into existing pairs."
            )
        # No grouped nodes at all -> full consecutive fallback.
        if not group_map:
            return self._consecutive_pairs(ranks)

        node_groups: List[Dict[int, NodeTopologyMeta]] = []
        inter_by_group: Dict[int, List[int]] = {}
        rng = random.Random(self._rdzv_round * 100003 + len(ranks))
        single_group = len(insertion_order) == 1

        for g in insertion_order:
            members = list(group_map[g])
            rng.shuffle(members)
            half = len(members) // 2
            intra_members = members[:half]
            inter_members = members[half:]
            # With a single group there is no "cross group"; pair everyone
            # intra so each node still gets a partner.
            if single_group:
                intra_members = members
                inter_members = []
            i = 0
            while i + 1 < len(intra_members):
                r0 = intra_members[i]
                r1 = intra_members[i + 1]
                node_groups.append(
                    {r0: self._rdzv_nodes[r0], r1: self._rdzv_nodes[r1]}
                )
                i += 2
            # Odd intra leftover is handed off to the inter pool.
            if i < len(intra_members):
                inter_members = inter_members + [intra_members[i]]
            if inter_members:
                inter_by_group[g] = inter_members

        inter_pairs = self._balanced_cross_group_pairs(inter_by_group)
        node_groups.extend(inter_pairs)

        # Fold any still-unmatched node (odd inter total, or ungrouped) into
        # an existing pair as an extra member, matching legacy leftovers.
        paired_ranks: set[int] = set()
        for grp in node_groups:
            paired_ranks.update(grp.keys())
        leftovers = [r for r in ranks if r not in paired_ranks]
        if leftovers:
            logger.info(
                f"network-check round 0: folding {len(leftovers)} leftover "
                f"node(s) (ranks={leftovers}) into existing pairs."
            )
            for r in leftovers:
                if node_groups:
                    node_groups[-1][r] = self._rdzv_nodes[r]
                else:
                    node_groups.append({r: self._rdzv_nodes[r]})

        intra_pair_count = len(node_groups) - len(inter_pairs)
        logger.info(
            f"network-check round {self._rdzv_round} coverage pairing "
            f"(mode=with_group): groups={len(insertion_order)}, "
            f"intra_pairs={intra_pair_count}, inter_pairs={len(inter_pairs)}, "
            f"total_pairs={len(node_groups)}, nodes={len(ranks)}, "
            f"ungrouped={len(ungrouped)}."
        )
        self._log_pairing_sample(node_groups, sample_per_kind=8)
        return node_groups

    def _balanced_cross_group_pairs(
        self, inter_by_group: Dict[int, List[int]]
    ) -> List[Dict[int, NodeTopologyMeta]]:
        """Pair inter-group nodes across different groups, balanced.

        Nodes are laid out grouped-by-group; each node at position i is
        paired with the node half-way across the layout ((i + n//2) mod n),
        which (with >=2 groups) lands in a different group; a linear scan
        fixes any same-group collision. Each group's inter nodes are thus
        spread across the groups ~half-way around, giving balanced
        cross-group coverage. Nodes with no available different-group
        partner are left for the caller to fold.
        """
        flat: List[Tuple[int, int]] = []  # (rank, group_idx)
        for g, ranks in inter_by_group.items():
            for r in ranks:
                flat.append((r, g))
        n = len(flat)
        if n < 2:
            return []
        offset = n // 2
        used = [False] * n
        pairs: List[Dict[int, NodeTopologyMeta]] = []
        for i in range(n):
            if used[i]:
                continue
            j = (i + offset) % n
            for _ in range(n):
                if j != i and not used[j] and flat[j][1] != flat[i][1]:
                    break
                j = (j + 1) % n
            if j == i or used[j] or flat[j][1] == flat[i][1]:
                continue  # no different-group partner; caller folds node i
            used[i] = True
            used[j] = True
            r0 = flat[i][0]
            r1 = flat[j][0]
            pairs.append({r0: self._rdzv_nodes[r0], r1: self._rdzv_nodes[r1]})
        return pairs

    def _log_pairing_sample(
        self, node_groups: List[Dict[int, NodeTopologyMeta]], sample_per_kind=8
    ):
        """Log a small intra/inter pairing sample for debugging."""
        intra: List[str] = []
        inter: List[str] = []
        for grp in node_groups:
            ranks_in_grp = list(grp.keys())
            if len(ranks_in_grp) < 2:
                continue
            r0, r1 = ranks_in_grp[0], ranks_in_grp[1]
            g0 = self._query_node_group(self._rdzv_nodes[r0].node_id)[0]
            g1 = self._query_node_group(self._rdzv_nodes[r1].node_id)[0]
            tag = "intra" if g0 >= 0 and g0 == g1 else "inter"
            entry = (
                f"r{r0}/n{self._rdzv_nodes[r0].node_id}(g{g0})<->"
                f"r{r1}/n{self._rdzv_nodes[r1].node_id}(g{g1})"
            )
            if tag == "intra":
                intra.append(entry)
            else:
                inter.append(entry)
        logger.info(
            f"network-check pairing sample [intra, first {sample_per_kind}]: "
            f"{intra[:sample_per_kind]}"
        )
        logger.info(
            f"network-check pairing sample [inter, first {sample_per_kind}]: "
            f"{inter[:sample_per_kind]}"
        )

    def _check_abnormal_nodes(self):
        abnormal_nodes = []
        normal_nodes = []
        for node_rank, status in self._node_status.items():
            if not self._rdzv_nodes or node_rank not in self._rdzv_nodes:
                logger.info(
                    "Skip check abnormal nodes due to rdzv manager "
                    "hasn't been initialized."
                )
                return
            node_id = self._rdzv_nodes[node_rank].node_id
            if status:
                normal_nodes.append(node_id)
            else:
                abnormal_nodes.append(node_id)
        logger.info(
            f"Normal nodes: {normal_nodes}.\nAbnormal nodes: {abnormal_nodes}"
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
                f"{self._name} round {self._rdzv_round}: The node status "
                f"are: {node_status}, "
                f"the node group are: {self._get_print_node_groups()}"
            )
            node_check_times = self._map_node_rank_to_id(self._node_times)
            logger.info(
                f"{self._name} round {self._rdzv_round}: The node elapsed time "
                f"are {node_check_times}"
            )
            if self._event_reporter:
                class_type = type(self).__name__
                self._event_reporter.report_network_check_completed(
                    self._network_check_evt,
                    event_type=EventReportConstants.TYPE_INFO,
                    instance=class_type,
                    action=EventReportConstants.ACTION_STOP,
                    node_status=f"{node_status}",
                    node_check_times=f"{node_check_times}",
                    node_groups=f"{self._get_print_node_groups()}",
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
        allgather. If succeeded, the round should be set to the multiples of 2.
        """
        with self._lock:
            if not self._rdzv_nodes:
                logger.warning(
                    "Skip check for rdzv_nodes hasn't been initialized."
                )
                return [], NetworkFailureReason.NO_INIT
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
                    logger.warning(
                        f"Fault nodes(rank:node_id) are: {fault_nodes}"
                    )
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
            if len(self._reported_nodes) < len(self._rdzv_nodes):
                reason = NetworkFailureReason.WAITING_NODE
            elif len(self._straggler_nodes) == 0:
                stragglers = self._detect_stragglers()
                if stragglers:
                    logger.warning(f"Straggler: {stragglers}.")
                self._straggler_nodes.update(stragglers)
            return list(self._straggler_nodes), reason

    def _detect_stragglers(self):
        """Detect whether there is the straggler in the job."""
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


def create_training_rdzv_manager() -> RendezvousManager:
    """Factory to create the training rendezvous manager.

    Use master job args via global context to select the implementation.
    Supported values: "base", "ucp". Default is "base".
    """
    rdzv_type = (
        Context.singleton_instance().training_elastic_mode or "base"
    ).lower()
    if rdzv_type == "base":
        return ElasticTrainingRendezvousManager()
    if rdzv_type == "ucp":
        return UcpRdzvManager()
    logger.warning(
        f"Unknown training rendezvous manager type '{rdzv_type}', "
        "falling back to ElasticTrainingRendezvousManager."
    )
    return ElasticTrainingRendezvousManager()
