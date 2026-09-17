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

import threading
from typing import Dict, List, Optional

from dlrover.python.common.constants import (
    NodeGroupStrategy,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger


class SoftGroupSchedule(object):
    """Schedule of unequally-sized node groups (--soft-group-affinity).

    Fully isolated from the plain ``--group-affinity`` path
    (:class:`NodeGroupSchedule`/:func:`validate_topology`/
    :func:`resolve_group_id` in ``dlrover.python.master.resource.job``):
    a soft schedule declares a heterogeneous per-group pod count and
    resolves every worker's group id at pod-creation time so the scaler
    labels pods with ``scheduling/rack-group`` for segment-affinity
    scheduling. There is deliberately NO equal-size constraint, but
    EVERY group size must be a multiple of the EP slot
    (``ep_workers = ETP*EP/R`` pods): this EP alignment is an
    unconditional start-up requirement, not an option.

    Args:
        strategy: one of :class:`NodeGroupStrategy` (ep_dp_pp or
            ep_pp_dp), consulted by :func:`resolve_soft_group_id`.
        sizes: ``{group_id: pod count}``, e.g. ``{0: 30, 1: 20}``; the
            keys are the node-group ids and can be any non-negative
            integers (the fill order is ascending) and every count must
            be a multiple of the EP slot.
        tp / pp / ep / etp / cp: Megatron parallel sizes. ``tp`` and
            ``cp`` are recorded but not used by the layout math today;
            an unset ``etp`` (None) inherits ``tp`` and the real EP
            group size is ``etp * ep``.
        num_nodes: total number of worker nodes (``N``), must equal the
            sum of ``sizes``.
        ranks_per_node: number of ranks per worker node (``R``).
        no_group_failover: when true (--no-group-failover), a relaunched
            (FO) worker drops its node-group labels so it can be
            scheduled onto any segment (see
            DistributedJobManager._relaunch_node); when false (default)
            the relaunch keeps the group labels and follows the
            original segment affinity.
    """

    def __init__(
        self,
        strategy: str,
        sizes: Dict[int, int],
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
        etp: Optional[int] = None,
        cp: int = 1,
        num_nodes: int = 0,
        ranks_per_node: int = 0,
        no_group_failover: bool = False,
    ):
        self.strategy = strategy
        self.sizes = sizes
        self.tp = tp
        self.pp = pp
        self.ep = ep
        # None (unset, --expert-tensor-parallel-size not configured)
        # inherits the recorded tensor_model_parallel_size.
        self.etp = etp if etp is not None else tp
        self.cp = cp
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.no_group_failover = no_group_failover
        # Laid-out {rank_index: group_id} for the ep_pp_dp strategy,
        # built lazily by resolve_soft_group_id.
        self._ep_pp_dp_layout: Optional[List[int]] = None
        self._layout_lock = threading.Lock()


def ep_group_workers(schedule: SoftGroupSchedule) -> int:
    """Return the EP slot size in pods (``ETP*EP/R``, one EP
    communication group of ``ETP*EP`` consecutive ranks). The caller must
    have validated ``ETP*EP % R == 0`` (see
    validate_soft_group_topology)."""
    ep_group = schedule.etp * schedule.ep
    if schedule.ranks_per_node <= 0:
        return 0
    return ep_group // schedule.ranks_per_node


def validate_soft_group_topology(
    schedule: Optional[SoftGroupSchedule],
) -> None:
    """Validate a soft group schedule (``--soft-group-affinity``).

    Distinct from :func:`validate_topology`: there is no equal-size
    constraint and no segment-count divisibility; raising ``ValueError``
    leaves the schedule unapplied so the master fails to start cleanly.

    Constraints (``N`` = #worker nodes, ``R`` = ranks/node, ``G`` =
    ``len(sizes)``, ``model_parallel = PP`` with TP and CP recorded but
    not used, ``EPG = ETP*EP`` the real EP group size (an unset ``etp``
    inherits the recorded ``tp``), ``dense_dp = N*R/PP``,
    ``ep_workers = EPG/R``):

      - ``sizes`` is non-empty with positive counts and non-negative keys;
      - ``sum(sizes) == N``;
      - ``PP > 0`` and ``EP > 0``;
      - ``N % PP == 0`` and ``dense_dp % EPG == 0`` so the
        data-parallel size ``dp = dense_dp/EPG`` is an integral number;
      - ``EPG % R == 0`` (an EP slot is composed of whole nodes) and
        every group size is a multiple of ``ep_workers`` — the EP
        alignment is mandatory by default, no opt-in flag;
      - when the strategy is ``ep_pp_dp``: additionally
        ``DP_nodes % ep_workers == 0`` so every pipeline stage holds a
        whole number of EP slots.
    """
    if schedule is None or not schedule.sizes:
        return

    sizes = schedule.sizes
    pp, ep, etp = schedule.pp, schedule.ep, schedule.etp
    n, r = schedule.num_nodes, schedule.ranks_per_node

    total = sum(sizes.values())
    if total != n:
        raise ValueError(
            "soft-group-affinity sizes sum to "
            f"{total} pods, but the job declares N={n} worker replicas; "
            "align --soft-group-affinity with the worker count."
        )
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(
            f"soft-group-affinity group sizes must be positive, got {sizes}."
        )
    if any(key < 0 for key in sizes.keys()):
        raise ValueError(
            "soft-group-affinity group ids must be non-negative, got "
            f"{sorted(sizes.keys())}."
        )

    if pp <= 0 or ep <= 0:
        raise ValueError(
            f"--soft-group-affinity requires pp={pp} and ep={ep} to be "
            "positive."
        )
    if r <= 0:
        raise ValueError(
            "--soft-group-affinity requires ranks_per_node>0 "
            f"(NodeResource.gpu_num), got {r}."
        )

    # TP and CP are recorded but not used; the layout math only involves
    # PP and the real EP group size ETP*EP.
    model_parallel = pp
    if n % model_parallel != 0:
        raise ValueError(
            "--soft-group-affinity requires the worker node count N="
            f"{n} to be divisible by PP={model_parallel}."
        )
    ep_group = etp * ep
    dense_dp = (n * r) // model_parallel
    if dense_dp % ep_group != 0:
        raise ValueError(
            "--soft-group-affinity requires dense_dp="
            f"{dense_dp} (N*R/PP) to be divisible by the EP group size "
            f"ETP*EP={ep_group} so the data-parallel size "
            "dp=dense_dp/(ETP*EP) is an integer."
        )

    # EP alignment is the default, not an option: an EP slot must be
    # well defined (EPG % R == 0) and every group must be a whole number
    # of EP slots, otherwise the master fails to start.
    if ep_group % r != 0:
        raise ValueError(
            "--soft-group-affinity requires the EP group size ETP*EP="
            f"{ep_group} to be divisible by ranks_per_node R={r} so an EP "
            "group (slot) is composed of whole nodes."
        )
    ep_workers = ep_group // r
    for group_id, size in sizes.items():
        if size % ep_workers != 0:
            raise ValueError(
                "--soft-group-affinity requires every group size to be a "
                f"multiple of the EP slot (ep_workers={ep_workers}); "
                f"group {group_id} has {size} pods."
            )

    if schedule.strategy == NodeGroupStrategy.EP_PP_DP:
        dp_nodes = n // model_parallel
        if dp_nodes % ep_workers != 0:
            raise ValueError(
                "--soft-group-affinity with node-group-strategy=ep_pp_dp "
                f"requires DP_nodes={dp_nodes} (nodes per pipeline stage) "
                f"to be divisible by ep_workers={ep_workers}."
            )


def _ep_pp_dp_layout(schedule: SoftGroupSchedule) -> List[int]:
    """Compute (once per schedule) the full-world group sequence of the
    ``ep_pp_dp`` strategy: ``{rank_index: group_id}`` for ``k = 0..N-1``.

    The layout packs whole EP slots into ep-pp COLUMNS. The world has
    ``C = DP_nodes / ep_workers`` slot positions (columns); a column is
    one slot position spanning ALL ``pp`` pipeline stages (``pp`` EP
    slots, i.e. one Megatron PP column). Each column is filled by a
    greedy: repeatedly take the group with the MOST remaining whole EP
    slots (ties broken by the smallest group id) and place
    ``min(remaining, column capacity)`` slots of it until the column is
    full, then move to the next column.

    Why the largest-remaining greedy (fragment-deferring FFD): a group
    with at least ``pp`` remaining slots fills whole columns alone, so
    its PP columns are intra-group — ZERO cross-segment PP hops. Only
    residual fragments (``s_g % pp != 0`` leftover slots) share a
    column with other fragments; one group switch inside a column is
    exactly one cross-segment PP hop. Deferring fragments to the tail
    lets them pair with each other, which reaches the theoretical
    minimum number of mixed columns for nearly every input: the
    odd-slot groups lower-bound the mixed columns and the greedy pairs
    the residuals together ({3, 3, 2} with pp=2 packs as [aa][bb][cc][ab]
    — one mixed column, while a descending chained fill pays two).

    The column inventory is global (``pp * C * ep_workers`` slot-pods,
    covered by ``sum(sizes)``), so unlike a per-stage rotation the
    packing never starves a late stage: {g1: 6, g2: 2, g3: 2} with
    pp=2 and 1-pod slots lays out fully, where a per-stage round robin
    drains the small groups in stage 0 and cannot fill stage 1.

    Validation guarantees every size is a whole number of EP slots, so
    the greedy covers exactly all pods; the crumb tail below is a
    defensive fallback for schedules built bypassing validation.
    """
    with schedule._layout_lock:
        if schedule._ep_pp_dp_layout is not None:
            return schedule._ep_pp_dp_layout
        sizes = schedule.sizes
        total = sum(sizes.values())
        slot = ep_group_workers(schedule)
        slot = slot if slot > 0 else 1
        # Whole EP slots remaining per group; sub-slot crumbs are kept
        # apart for the defensive tail (empty for validated schedules).
        remaining: Dict[int, int] = {}
        crumbs: Dict[int, int] = {}
        for group_id in sorted(sizes):
            if sizes[group_id] > 0:
                remaining[group_id] = sizes[group_id] // slot
                crumbs[group_id] = sizes[group_id] % slot

        # TP and CP are recorded but not used; nodes per pipeline stage
        # is purely N/PP.
        pp = max(schedule.pp, 1)
        dp_nodes = total // pp
        # Number of slot positions per stage == the number of PP columns.
        columns = dp_nodes // slot

        # Greedy column packing: one column of pp slots at a time, every
        # pick taking min(remaining of the largest group, capacity) slots.
        col_slots: List[List[int]] = []
        for _col in range(columns):
            groups: List[int] = []
            while len(groups) < pp and any(v > 0 for v in remaining.values()):
                group_id = max(
                    sorted(remaining.keys()),
                    key=lambda k: (remaining[k], -k),
                )
                take = min(remaining[group_id], pp - len(groups))
                groups.extend([group_id] * take)
                remaining[group_id] -= take
            col_slots.append(groups)

        # Flatten the columns into the stage-major pod sequence: the
        # s-th pipeline stage consumes the s-th slot of every column.
        layout: List[int] = []
        for stage in range(pp):
            for slot_groups in col_slots:
                if stage < len(slot_groups):
                    layout.extend([slot_groups[stage]] * slot)

        # Defensive crumb tail (bypassed validation only): append the
        # sub-slot pods pod by pod in ascending group order.
        missing = total - len(layout)
        if missing > 0 and any(v > 0 for v in crumbs.values()):
            logger.warning(
                "The ep_pp_dp soft group layout found %d sub-slot pod(s) "
                "(the schedule bypassed the EP alignment validation); "
                "they are appended pod by pod and their EP groups may "
                "straddle.",
                missing,
            )
            for group_id in sorted(crumbs.keys()):
                for _ in range(crumbs[group_id]):
                    if len(layout) < total:
                        layout.append(group_id)

        if len(layout) != total:
            raise ValueError(
                "soft-group-affinity failed to lay out all worker pods; "
                f"expected {total}, got {len(layout)}."
            )

        straddled = [
            start
            for start in range(0, total, slot)
            if len(set(layout[start : start + slot])) > 1
        ]
        if straddled:
            logger.warning(
                "The ep_pp_dp soft group layout contains %d EP slot(s) "
                "crossing groups (sizes %s), starting at pod %s.",
                len(straddled),
                sizes,
                straddled[:5],
            )
        else:
            logger.info(
                "The ep_pp_dp soft group layout of sizes %s keeps every "
                "EP slot inside one group.",
                sizes,
            )
        schedule._ep_pp_dp_layout = layout
        return layout


def resolve_soft_group_id(
    schedule: Optional[SoftGroupSchedule],
    node_type: str,
    rank_index: int,
) -> Optional[int]:
    """Resolve the soft node group id of a worker by its rank index.

    - ``ep_dp_pp``: groups occupy cumulative contiguous rank ranges in
      ascending group-id order (fill up one group then the next).
    - ``ep_pp_dp``: the EP-column greedy packing layout (see
      :func:`_ep_pp_dp_layout`).

    Returns ``None`` for non-worker nodes, when no soft schedule is set,
    or when the rank falls beyond the declared world (e.g. a scale-up
    beyond ``sum(sizes)``), leaving the pod unlabeled.
    """
    if schedule is None or not schedule.sizes:
        return None
    if node_type != NodeType.WORKER:
        return None
    total = sum(schedule.sizes.values())
    if rank_index < 0 or rank_index >= total:
        return None
    if schedule.strategy == NodeGroupStrategy.EP_PP_DP:
        return _ep_pp_dp_layout(schedule)[rank_index]

    start = 0
    for group_id in sorted(schedule.sizes.keys()):
        size = schedule.sizes[group_id]
        if start <= rank_index < start + size:
            return group_id
        start += size
    return None
