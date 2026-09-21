# Copyright 2026 The DLRover Authors. All rights reserved.
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

import json
import os
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

from dlrover.python.common.constants import NodeGroupStrategy
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.elastic_training.net_topology import (
    NodeTopologyMeta,
    TopologyQuerier,
    TopologySorter,
)

# The cluster topology for --enable-topology-rerank: a JSON dict
# {node_ip: psw} provided either inline by the DLROVER_GROUP_TOPOLOGY
# env var or as a file at the path in DLROVER_GROUP_TOPOLOGY_FILE (the
# inline env takes precedence). The platform (e.g. the dlrover-addon
# distribution) is responsible for feeding the business cluster
# topology here.
GROUP_TOPOLOGY_ENV = "DLROVER_GROUP_TOPOLOGY"
GROUP_TOPOLOGY_FILE_ENV = "DLROVER_GROUP_TOPOLOGY_FILE"

# The log prefix of every topology-rerank message so they are greppable
# in the master and agent logs.
RERANK_LOG_PREFIX = "RERANK"


class GroupTopologyQuerier(TopologyQuerier):
    """Query the psw (physical switch) group of a node from the cluster
    topology configured by the ``DLROVER_GROUP_TOPOLOGY`` env var or the
    ``DLROVER_GROUP_TOPOLOGY_FILE`` file.

    Used by the master when ``--enable-topology-rerank`` is configured.
    Nodes whose IP is not in the map report an empty psw and fall into
    the rerank's DEFAULT group, a peer of the known groups.
    """

    def __init__(self):
        self._ip_to_psw: Dict[str, str] = _load_group_topology()

    def query(self, node_ip) -> Tuple[str, str]:
        return "", self._ip_to_psw.get(node_ip, "")


def _load_group_topology() -> Dict[str, str]:
    raw = os.getenv(GROUP_TOPOLOGY_ENV, "")
    source = f"env {GROUP_TOPOLOGY_ENV}"
    if not raw:
        topology_file = os.getenv(GROUP_TOPOLOGY_FILE_ENV, "")
        if topology_file:
            source = f"file {topology_file}"
            with open(topology_file, "r") as f:
                raw = f.read()
    if not raw:
        logger.warning(
            "%s: no cluster topology found (neither the %s env var nor "
            "the %s file); every node reports an empty psw and the "
            "topology rerank keeps the rendezvous order.",
            RERANK_LOG_PREFIX,
            GROUP_TOPOLOGY_ENV,
            GROUP_TOPOLOGY_FILE_ENV,
        )
        return {}
    try:
        topology = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid cluster topology JSON from the {source}: {e}"
        ) from e
    if not isinstance(topology, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in topology.items()
    ):
        raise ValueError(
            f"The cluster topology from the {source} must be a JSON dict "
            "mapping node IPs to psw strings, e.g. "
            '{"33.167.120.39": "SG2"}.'
        )
    counts: Dict[str, int] = {}
    for psw in topology.values():
        counts[psw] = counts.get(psw, 0) + 1
    logger.info(
        "%s: GroupTopologyQuerier loaded %d node(s) in %d psw group(s) "
        "from the %s: %s.",
        RERANK_LOG_PREFIX,
        len(topology),
        len(counts),
        source,
        dict(sorted(counts.items())),
    )
    return topology


def _ep_dp_pp_group_sequence(sizes: Dict[int, int]) -> List[int]:
    """Contiguous group sequence for the ep_dp_pp strategy: one group's
    slots directly after another in ascending group-id order."""
    sequence: List[int] = []
    for group_id in sorted(sizes.keys()):
        sequence.extend([group_id] * sizes[group_id])
    return sequence


def _ep_pp_dp_group_sequence(
    sizes: Dict[int, int], pp: int, ep_workers: int
) -> List[int]:
    """Greedy ep-pp column packing of the psw groups for the ep_pp_dp
    strategy; returns the group id of every world slot (one node per
    slot), len == sum(sizes).

    The world is cut into COLUMNS: a column holds ``pp`` EP slots and
    spans all the pipeline stages of one Megatron PP column, and an EP
    slot is ``ep_workers`` consecutive nodes of one EP communication
    group. Each column is filled by a greedy that repeatedly takes slots
    from the group with the MOST remaining whole slots (ties broken by
    the smallest group id): a group with at least ``pp`` remaining slots
    fills whole columns alone so its PP stages never leave the psw, and
    only residual fragments share a column. The columns are flattened
    stage-major: pipeline stage ``s`` consumes the s-th slot of every
    column.

    Unlike the --soft-group-affinity path there is NO alignment
    constraint here: the group sizes are the observed per-psw node
    counts, so leftover whole slots (``N % pp`` and
    ``DP_nodes % ep_workers`` remainders) are appended slot by slot
    after the columns and sub-slot crumbs pod by pod after that. Such
    tail slots may straddle psw groups, which the caller reports.
    """
    total = sum(sizes.values())
    pp = max(pp, 1)
    ep_workers = max(ep_workers, 1)
    group_ids = sorted(gid for gid, size in sizes.items() if size > 0)
    remaining = {gid: sizes[gid] // ep_workers for gid in group_ids}
    crumbs = {gid: sizes[gid] % ep_workers for gid in group_ids}

    columns = (total // pp) // ep_workers
    column_slots: List[List[int]] = []
    for _ in range(columns):
        slots: List[int] = []
        while len(slots) < pp and any(v > 0 for v in remaining.values()):
            group_id = max(
                sorted(remaining.keys()),
                key=lambda k: (remaining[k], -k),
            )
            take = min(remaining[group_id], pp - len(slots))
            slots.extend([group_id] * take)
            remaining[group_id] -= take
        column_slots.append(slots)

    sequence: List[int] = []
    for stage in range(pp):
        for slots in column_slots:
            if stage < len(slots):
                sequence.extend([slots[stage]] * ep_workers)

    # Leftover whole EP slots the column inventory could not place.
    for group_id in group_ids:
        while remaining[group_id] > 0 and len(sequence) + ep_workers <= total:
            sequence.extend([group_id] * ep_workers)
            remaining[group_id] -= 1
    # Sub-slot crumbs, pod by pod.
    for group_id in group_ids:
        for _ in range(crumbs[group_id]):
            if len(sequence) < total:
                sequence.append(group_id)

    if len(sequence) != total:
        raise ValueError(
            "topology rerank failed to lay out all nodes: expected "
            f"{total}, got {len(sequence)}."
        )
    return sequence


class GroupTopologySorter(TopologySorter):
    """Reorder the elastic-training rendezvous world by the psw groups
    observed at join time (``--enable-topology-rerank``).

    Each distinct psw becomes one node group of ARBITRARY size (no
    EP-slot alignment constraint, unlike --soft-group-affinity), laid
    out by the NodeGroupStrategy:

    - ``ep_dp_pp``: the groups occupy contiguous world ranges in
      ascending psw order;
    - ``ep_pp_dp``: EP slots are greedily packed into ep-pp columns so
      EP groups and PP stay intra-psw as much as possible.

    A node whose IP is not in the cluster topology reports an empty
    psw and joins the DEFAULT group, a PEER of the known groups: it is
    laid out under the same rule ("" sorts first in the ascending-psw
    group order) instead of being pushed to the tail.

    Every layout slot is then instantiated with a concrete node taken
    from the group in ascending node_rank order, so the same metas
    always produce the same world order (deterministic and idempotent
    across rendezvous rounds).

    Only the ORDER of the world dict changes: the keys (node_rank, the
    stable pod identity) are preserved, which is enough to rerank every
    worker process because agents derive all global ranks from the
    world order. Every log line is prefixed with "RERANK" so the whole
    rerank is greppable in the master log.
    """

    def __init__(
        self,
        strategy: str,
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
        etp: Optional[int] = None,
        cp: int = 1,
        ranks_per_node: Optional[int] = None,
    ):
        self.strategy = strategy
        self.tp = tp
        self.pp = pp
        self.ep = ep
        self.etp = etp
        self.cp = cp
        self.ranks_per_node = ranks_per_node

    def sort(
        self, nodes: Dict[int, NodeTopologyMeta]
    ) -> Dict[int, NodeTopologyMeta]:
        """Reorder ``nodes`` (keyed by node_rank) into the psw-aware
        world order; the keys and values are preserved, only the key
        order changes."""
        if not nodes:
            return nodes
        if self.strategy not in (
            NodeGroupStrategy.EP_DP_PP,
            NodeGroupStrategy.EP_PP_DP,
        ):
            logger.warning(
                "%s: unknown node-group strategy %r; keep the "
                "rendezvous order unchanged.",
                RERANK_LOG_PREFIX,
                self.strategy,
            )
            return nodes

        # One node group per distinct psw. A node whose IP is not in
        # the cluster topology reports an empty psw and joins the
        # DEFAULT group, a peer of the known groups: group ids follow
        # the ascending-psw order ("" sorts first), with no special
        # tail placement.
        psws = {meta.psw for meta in nodes.values()}
        psw_group_ids = {psw: gid for gid, psw in enumerate(sorted(psws))}

        group_metas: Dict[int, List[NodeTopologyMeta]] = {}
        for meta in nodes.values():
            group_metas.setdefault(psw_group_ids[meta.psw], []).append(meta)
        group_ids = sorted(group_metas)
        if len(group_ids) <= 1:
            logger.info(
                "%s: skip the reorder because only %d psw group(s) is "
                "detected (psw=%s); keep the rendezvous order.",
                RERANK_LOG_PREFIX,
                len(group_ids),
                sorted(psws),
            )
            return nodes

        for group_id in group_ids:
            group_metas[group_id].sort(key=lambda meta: meta.node_rank)
        sizes = {
            group_id: len(group_metas[group_id]) for group_id in group_ids
        }
        total = len(nodes)

        # Prefer the configured ranks_per_node; fall back to the local
        # world size observed at the rendezvous (uniform across nodes).
        ranks_per_node = self.ranks_per_node
        observed = next(iter(nodes.values())).process_num
        if not ranks_per_node or ranks_per_node <= 0:
            ranks_per_node = observed
        if ranks_per_node <= 0:
            logger.warning(
                "%s: skip the reorder because ranks_per_node is "
                "unavailable (configured=%s, observed=%s); keep the "
                "rendezvous order.",
                RERANK_LOG_PREFIX,
                self.ranks_per_node,
                observed,
            )
            return nodes
        if (
            self.ranks_per_node
            and observed
            and self.ranks_per_node != observed
        ):
            logger.warning(
                "%s: the configured ranks_per_node=%s differs from the "
                "observed local world size=%s; the layout uses the "
                "observed one.",
                RERANK_LOG_PREFIX,
                self.ranks_per_node,
                observed,
            )
            ranks_per_node = observed

        etp = self.etp if self.etp is not None else self.tp
        ep_group = etp * self.ep
        ep_workers = ep_group // ranks_per_node
        if ep_workers <= 0:
            ep_workers = 1
        if ep_group % ranks_per_node != 0:
            logger.warning(
                "%s: the EP group size ETP*EP=%d is not divisible by "
                "ranks_per_node=%d; fall back to ep_workers=%d and the "
                "EP slots may straddle psw groups.",
                RERANK_LOG_PREFIX,
                ep_group,
                ranks_per_node,
                ep_workers,
            )

        group_psws = {gid: psw for psw, gid in psw_group_ids.items()}
        logger.info(
            "%s: config strategy=%s, tp=%d, pp=%d, ep=%d, etp=%d, cp=%d, "
            "ranks_per_node=%d, nodes=%d, groups=%d, ep_workers=%d.",
            RERANK_LOG_PREFIX,
            self.strategy,
            self.tp,
            self.pp,
            self.ep,
            etp,
            self.cp,
            ranks_per_node,
            total,
            len(group_ids),
            ep_workers,
        )
        for group_id in group_ids:
            ranks = [meta.node_rank for meta in group_metas[group_id]]
            logger.info(
                "%s: group psw=%s size=%d node_ranks=%s.",
                RERANK_LOG_PREFIX,
                group_psws[group_id] or "<default>",
                sizes[group_id],
                ranks,
            )

        if self.strategy == NodeGroupStrategy.EP_PP_DP:
            sequence = _ep_pp_dp_group_sequence(sizes, self.pp, ep_workers)
        else:
            sequence = _ep_dp_pp_group_sequence(sizes)

        straddled = [
            start
            for start in range(0, total, ep_workers)
            if len(set(sequence[start : start + ep_workers])) > 1
        ]
        if straddled:
            logger.warning(
                "%s: %d EP slot(s) straddle psw groups (sizes=%s), "
                "starting at node %s.",
                RERANK_LOG_PREFIX,
                len(straddled),
                sizes,
                straddled[:5],
            )

        # Instantiate every slot with a concrete node of the group, in
        # ascending node_rank order, so the layout is deterministic.
        node_queues = {
            group_id: deque(meta.node_rank for meta in group_metas[group_id])
            for group_id in group_ids
        }
        ordered_ranks: List[int] = []
        for group_id in sequence:
            if node_queues[group_id]:
                ordered_ranks.append(node_queues[group_id].popleft())
        # Defensive: append any node the sequence missed (the layout is
        # expected to cover exactly all nodes).
        for group_id in group_ids:
            while node_queues[group_id]:
                ordered_ranks.append(node_queues[group_id].popleft())

        old_positions = {
            node_rank: idx for idx, node_rank in enumerate(nodes.keys())
        }
        sorted_nodes: Dict[int, NodeTopologyMeta] = OrderedDict()
        changed = 0
        for new_position, node_rank in enumerate(ordered_ranks):
            meta = nodes[node_rank]
            sorted_nodes[node_rank] = meta
            if new_position != old_positions[node_rank]:
                changed += 1
            logger.info(
                "%s: node node_rank=%s ip=%s psw=%s world_rank %s->%s.",
                RERANK_LOG_PREFIX,
                node_rank,
                meta.node_ip,
                meta.psw or "<default>",
                old_positions[node_rank],
                new_position,
            )
        logger.info(
            "%s: order: world position -> node_rank: %s; %d/%d nodes moved.",
            RERANK_LOG_PREFIX,
            ordered_ranks,
            changed,
            total,
        )
        return sorted_nodes
