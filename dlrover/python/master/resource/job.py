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
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

from dlrover.python.brain.client import GlobalBrainClient
from dlrover.python.common.constants import (
    JobOptStage,
    NodeGroupStrategy,
    NodeResourceLimit,
    NodeType,
    OptimizeMode,
    OptimizeWorkerPhase,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.common.serialize import JsonSerializable
from dlrover.python.master.resource.brain_optimizer import (
    BrainResoureOptimizer,
)
from dlrover.python.master.resource.local_optimizer import PSLocalOptimizer
from dlrover.python.master.resource.optimizer import (
    ResourcePlan,
    SimpleOptimizer,
)
from dlrover.python.scheduler.job import ResourceLimits
from dlrover.python.util.args_util import validate_group_affinity_equal_size

_WORKER_OPTIMIZE_PHASE = "optimizer.worker.optimize-phase"

_dlrover_context = Context.singleton_instance()


def new_ps_resource_optimizer(
    optimize_mode: str, job_uuid, resource_limits: ResourceLimits
):
    logger.info(
        "New %s resource optimizer for job %s", optimize_mode, job_uuid
    )
    if optimize_mode == OptimizeMode.CLUSTER:
        if GlobalBrainClient.BRAIN_CLIENT.available():
            return BrainResoureOptimizer(job_uuid, resource_limits)
        else:
            logger.warning(
                "Brain service is not available, use a local optimizer"
            )
            return PSLocalOptimizer(job_uuid, resource_limits)
    elif optimize_mode == OptimizeMode.SINGLE_JOB:
        return PSLocalOptimizer(job_uuid, resource_limits)
    else:
        logger.warning(
            "Not support optimization mode %s, use a simple optimizer",
            optimize_mode,
        )
        return SimpleOptimizer(job_uuid, resource_limits)


class NodeGroupSchedule(object):
    """Strategies + parallel sizes used to map a worker node (ranked by its
    creation order, i.e. its ``rank_index`` at the node granularity) into a
    node group. A node group corresponds to a physical scheduling segment.

    Only ``strategy`` is meaningful by itself. The parallel sizes are used by
    the :func:`validate_topology` and :func:`_resolve_stripe_group_id` helpers
    when ``strategy == NodeGroupStrategy.EP_PP_DP``.

    Args:
        strategy: one of :class:`NodeGroupStrategy`. Defaults to
            ``CONTIGUOUS`` (current behavior).
        tp / pp / ep / cp: Megatron tensor/pipeline/expert/context parallel
            sizes. Only ``tp == 1`` and ``cp == 1`` are supported today.
        num_nodes: total number of worker nodes (``N``).
        ranks_per_node: number of ranks per worker node (``R``), i.e.
            ``NodeResource.gpu_num``.
    """

    def __init__(
        self,
        strategy: str = NodeGroupStrategy.CONTIGUOUS,
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
        cp: int = 1,
        num_nodes: int = 0,
        ranks_per_node: int = 0,
    ):
        self.strategy = strategy
        self.tp = tp
        self.pp = pp
        self.ep = ep
        self.cp = cp
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node


def _resolve_stripe_group_id(
    group_affinity: Dict[int, int],
    rank_index: int,
    schedule: NodeGroupSchedule,
) -> int:
    """Resolve the node group when ``strategy == ep_pp_dp``.

    One pipeline stage (a dense-DP group, ``dense_dp`` ranks) is striped
    across all groups/segments so that EP groups and PP stay intra-segment,
    while only the dense DP collective crosses segments. Working at the
    node granularity (``rank_index`` = node creation order):

        DP_nodes = N / (TP*PP*CP)          # nodes per pipeline stage
        seg      = DP_nodes / G            # nodes per segment per stage
        group(k) = (k % DP_nodes) // seg

    The caller is responsible for having validated the topology via
    :func:`validate_topology`, which guarantees the divisions are integral
    and ``seg >= 1``.
    """
    g = len(group_affinity)
    dp_nodes = schedule.num_nodes // (schedule.tp * schedule.pp * schedule.cp)
    seg = dp_nodes // g
    return (rank_index % dp_nodes) // seg


def resolve_group_id(
    group_affinity: Optional[Dict[int, int]],
    node_type: str,
    rank_index: int,
    schedule: Optional[NodeGroupSchedule] = None,
) -> Optional[int]:
    """Resolve the node group id for a node by its rank index.

    Only WORKER nodes are partitioned into node groups.

    - With the default ``schedule`` (None) or its strategy being
      :data:`NodeGroupStrategy.CONTIGUOUS`, groups are laid out in ascending
      group-id order, each occupying a contiguous ``[start, end)`` rank range
      sized by ``group_affinity[group_id]``. This is the existing behavior.
    - With strategy :data:`NodeGroupStrategy.EP_PP_DP`, the stripe mapping in
      :func:`_resolve_stripe_group_id` is used instead. This requires a
      validated topology (see :func:`validate_topology`).

    Returns ``None`` when group affinity is not configured or the rank
    falls outside the declared worker range.
    """
    if node_type != NodeType.WORKER or not group_affinity:
        return None
    if (
        schedule is not None
        and schedule.strategy == NodeGroupStrategy.EP_PP_DP
    ):
        return _resolve_stripe_group_id(group_affinity, rank_index, schedule)
    ordered_groups: List[int] = sorted(group_affinity.keys())
    start = 0
    for group_id in ordered_groups:
        size = group_affinity[group_id]
        if start <= rank_index < start + size:
            return group_id
        start += size
    return None


def validate_topology(
    group_affinity: Optional[Dict[int, int]],
    schedule: Optional[NodeGroupSchedule],
) -> None:
    """Validate ``group_affinity`` against the parallel topology when the
    ``ep_pp_dp`` node-group strategy is enabled. Raises ``ValueError`` (and
    leaves ``group_affinity`` unapplied) whenever any constraint is violated.

    Constraints (``N`` = #worker nodes, ``R`` = ranks/node, ``G`` =
    ``len(group_affinity)``, ``dense_dp = N*R / (TP*PP*CP)``,
    ``S = dense_dp / G`` ranks per segment per stage):

      - scope: ``TP == 1`` and ``CP == 1``;
      - A: ``dense_dp % G == 0`` (each segment gets an integer #ranks/stage);
      - B: ``S % R == 0`` equiv ``N % (TP*PP*CP*G) == 0`` (a node's ``R``
            ranks do not cross a segment-per-stage block, and nodes/stages
            are whole);
      - C: ``S % EP == 0`` (an EP group does not cross a segment, which also
            makes ``de`` divisible across the ``G`` segments);
      - D: ``N % G == 0`` and all ``group_affinity`` values are equal to
            ``N/G`` (equal-size groups, the platform hard requirement), and
            the group ids are exactly ``{0, 1, ..., G-1}``;
      - E: ``EP % R == 0`` (an EP group is composed of whole nodes);
      - implicit: ``dense_dp`` and ``dp_nodes`` are integral and ``seg >= 1``.
    """
    if schedule is None or schedule.strategy != NodeGroupStrategy.EP_PP_DP:
        return
    if not group_affinity:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires --group-affinity to be set"
        )

    g = len(group_affinity)
    tp, pp, ep, cp = schedule.tp, schedule.pp, schedule.ep, schedule.cp
    n, r = schedule.num_nodes, schedule.ranks_per_node

    if tp != 1:
        raise ValueError(
            "node-group-strategy=ep_pp_dp currently requires TP=1 "
            f"(got tp={tp}); TP>1 support is planned for a later phase."
        )
    if cp != 1:
        raise ValueError(
            "node-group-strategy=ep_pp_dp currently requires CP=1 "
            f"(got cp={cp}); CP>1 support is planned for a later phase."
        )
    if r <= 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires ranks_per_node>0 "
            f"(NodeResource.gpu_num), got {r}."
        )

    model_parallel = tp * pp * cp
    # DP_nodes = N / (TP*PP*CP) must be whole; this also guarantees dense_dp
    # (= N*R/(TP*PP*CP)) is integral since R is integer.
    if n % model_parallel != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires the worker node count N "
            f"to be divisible by TP*PP*CP={model_parallel}, got N={n}."
        )

    # Constraint A: dense_dp is divisible by G (each segment gets an integer
    # number of ranks per pipeline stage).
    dense_dp = (n * r) // model_parallel
    if dense_dp % g != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires dense_dp="
            f"{dense_dp} (N*R/(TP*PP*CP)) to be divisible by G={g} segments."
        )
    s = dense_dp // g

    # Constraint B: a node's R ranks do not straddle a segment-per-stage block;
    # equivalently the nodes per segment per stage are integral
    # (N % (TP*PP*CP*G) == 0).
    if s % r != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires S=dense_dp/G={s} ranks "
            f"per segment per stage to be divisible by ranks_per_node R={r} "
            "(equivalently N % (TP*PP*CP*G) == 0)."
        )
    seg_nodes = s // r
    if seg_nodes < 1:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires at least one node per "
            f"segment per stage, got {seg_nodes}."
        )

    # Constraint C: an EP group fits and aligns inside a segment-per-stage
    # block (also guarantees de=S/EP is integral across the G segments).
    if ep <= 0:
        raise ValueError(
            f"node-group-strategy=ep_pp_dp requires EP>0, got ep={ep}."
        )
    if s % ep != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires the EP size "
            f"({ep}) to divide S=dense_dp/G={s} so each EP group stays "
            "inside a segment."
        )

    # Constraint E: an EP group is composed of whole nodes.
    if ep % r != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires EP="
            f"{ep} to be divisible by ranks_per_node R={r} so an EP "
            "group spans whole nodes."
        )

    # Constraint D: equal-size groups, group ids == {0..G-1}, each == N/G.
    if n % g != 0:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires the worker node count "
            f"N={n} to be divisible by G={g} segments (equal group sizes)."
        )
    expected_size = n // g
    # Raises ValueError when the group sizes are not all equal.
    equal_size = validate_group_affinity_equal_size(group_affinity)
    if equal_size != expected_size:
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires all group_affinity sizes "
            f"to be equal to N/G={expected_size}, got {equal_size}."
        )
    keys = sorted(group_affinity.keys())
    if keys != list(range(g)):
        raise ValueError(
            "node-group-strategy=ep_pp_dp requires contiguous group ids "
            f"{{0,1,...,{g - 1}}}, got {keys}."
        )


class JobResource(JsonSerializable):
    def __init__(self):
        self.node_group_resources: Dict[str, NodeGroupResource] = {}
        self.group_affinity: Optional[Dict[int, int]] = None
        # Node-group schedule (strategy + parallel sizes). Defaults to None,
        # i.e. ``CONTIGUOUS`` behavior; set to a NodeGroupSchedule(strategy=
        # ep_pp_dp, ...) when the ep_pp_dp strategy is enabled and validated.
        self.node_group_schedule: Optional[NodeGroupSchedule] = None

    def get_node_group_resource(self, node_type):
        return self.node_group_resources.get(node_type, None)

    def _get_group_node_num(self, node_type):
        if node_type in self.node_group_resources:
            return self.node_group_resources[node_type].count
        return 0

    def get_node_types(self):
        return list(self.node_group_resources.keys())

    def update_node_group_resource(self, node_type, num, cpu, memory):
        self.node_group_resources.setdefault(
            node_type,
            NodeGroupResource(
                count=0,
                node_resource=NodeResource(0, 0),
            ),
        )
        resource = self.node_group_resources[node_type]
        resource.count = num or resource.count
        resource.node_resource.cpu = cpu or resource.node_resource.cpu
        resource.node_resource.memory = memory or resource.node_resource.memory

    @property
    def worker_num(self):
        return self._get_group_node_num(NodeType.WORKER)

    @property
    def ps_num(self):
        return self._get_group_node_num(NodeType.PS)

    @property
    def evaluator_num(self):
        return self._get_group_node_num(NodeType.EVALUATOR)

    @property
    def chief_num(self):
        return self._get_group_node_num(NodeType.CHIEF)

    def init_job_node_meta(
        self,
        relaunch_on_worker_failure,
        service_create_fn,
        new_node_name_fn,
    ):
        """
        relaunch_on_worker_failure: int, the number of relaunches.
        service_create_fn: a callable function to create the name for a sevice.
        new_node_name_fn: a callable function to create the name for a node.
        return: a dict with pod_type as key, and another dict as value.
                The other dict uses pod id as key, and PodInfo as value.
        """
        job_nodes: Dict[str, Dict[int, Node]] = {}
        for node_type in self.get_node_types():
            group_resource = self.get_node_group_resource(node_type)
            config_resource = group_resource.node_resource
            group_nodes: Dict[int, Node] = {}
            group_size = self._group_count()
            for i in range(group_resource.count):
                group_id = self._resolve_group_id(node_type, i)
                group_nodes[i] = Node(
                    node_type=node_type,
                    node_id=i,
                    rank_index=i,
                    name=new_node_name_fn(node_type, i),
                    config_resource=copy.deepcopy(config_resource),
                    max_relaunch_count=relaunch_on_worker_failure,
                    service_addr=service_create_fn(node_type, i),
                    node_group=group_id,
                    node_group_size=group_size
                    if group_id is not None
                    else None,
                    node_group_id=group_id,
                )
            job_nodes[node_type] = group_nodes
        logger.info(
            "after initializing job node meta job_nodes are %s" % job_nodes
        )
        return job_nodes

    def _group_count(self) -> Optional[int]:
        """Return the total number of node groups, or None if group
        affinity is not configured."""
        if not self.group_affinity:
            return None
        return len(self.group_affinity)

    def _resolve_group_id(
        self, node_type: str, rank_index: int
    ) -> Optional[int]:
        """Resolve the node group id for a node by its rank index.

        Only WORKER nodes are partitioned into node groups. The groups are
        laid out according to ``self.node_group_schedule``: by default
        (None / contiguous) this is the ascending-group-id contiguous
        [start, end) rank range sized by ``group_affinity[group_id]``; when an
        ep_pp_dp schedule is configured, the stripe mapping is used.
        Returns None when group affinity is not configured.
        """
        return resolve_group_id(
            self.group_affinity,
            node_type,
            rank_index,
            self.node_group_schedule,
        )

    def adjust_worker_for_estimator(self):
        if (
            NodeType.CHIEF in self.node_group_resources
            and self.node_group_resources[NodeType.CHIEF].count > 0
        ) or (NodeType.WORKER not in self.node_group_resources):
            return

        worker = self.node_group_resources[NodeType.WORKER]
        if worker.count <= 0:
            return

        chief = self.node_group_resources.get(
            NodeType.CHIEF, NodeGroupResource.new_empty()
        )
        chief.count = 1
        chief.node_resource.cpu = worker.node_resource.cpu
        chief.node_resource.memory = worker.node_resource.memory
        self.node_group_resources[NodeType.CHIEF] = chief
        worker.count -= 1
        logger.info("self = %s", self.to_json())


class JobResourceOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def update_job_uuid(self, job_uuid):
        pass

    @abstractmethod
    def init_job_resource(self, job_resource: JobResource):
        """Initialize resource configuration for a job."""
        pass

    @abstractmethod
    def get_job_resource_plan(self) -> ResourcePlan:
        """Get resource plan for a job."""
        pass

    @abstractmethod
    def adjust_oom_resource(self, node: Node):
        """Adjust the resource configuration for OOM nodes"""
        pass

    @abstractmethod
    def get_config_resource(self) -> JobResource:
        pass


class PSJobResourceOptimizer(JobResourceOptimizer):
    """It generates resource configuration for a PS job."""

    def __init__(
        self,
        worker_resource: NodeGroupResource,
        ps_resource: NodeGroupResource,
        optimize_mode: str,
        job_uuid="",
        resource_limits=ResourceLimits(),
    ):
        self._worker_resource = worker_resource
        self._ps_resource = ps_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._original_ps_resource = copy.deepcopy(self._ps_resource)
        self._resource_optimizer = new_ps_resource_optimizer(
            optimize_mode, job_uuid, resource_limits
        )
        self._lock = threading.Lock()
        self.optimized_ps_mem = False
        self.optimize_worker_sampled = False
        self._job_stage = JobOptStage.CREATE
        self._last_ps_change_time = 0.0

    def set_job_stage(self, stage):
        self._job_stage = stage

    def get_job_stage(self):
        return self._job_stage

    def get_config_resource(self):
        job_config = JobResource()
        worker_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.WORKER] = worker_config
        ps_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.PS] = ps_config
        return job_config

    def update_job_uuid(self, job_uuid):
        self._resource_optimizer.update_job_uuid(job_uuid)

    def _init_job_resource_by_optimizer(self):
        plan = self._resource_optimizer.generate_opt_plan(self._job_stage)
        if not plan or plan.empty():
            logger.info("Use the default plan to start the job")
            plan = self._gen_default_resource_plan()
        self._job_stage = JobOptStage.WORKER_INITIAL

        if (
            _dlrover_context.auto_worker_enabled
            and NodeType.WORKER in plan.node_group_resources
        ):
            worker_resource = self._check_ignore_original_worker_resource(
                plan.node_group_resources[NodeType.WORKER]
            )
            self._worker_resource.update(
                worker_resource.count,
                worker_resource.node_resource.cpu,
                worker_resource.node_resource.memory,
            )
        if (
            _dlrover_context.auto_ps_enabled
            and NodeType.PS in plan.node_group_resources
        ):
            ps_resource = self._check_ignore_original_ps_resource(
                plan.node_group_resources[NodeType.PS]
            )
            self._ps_resource.update(
                ps_resource.count,
                ps_resource.node_resource.cpu,
                ps_resource.node_resource.memory,
            )

    def _gen_default_resource_plan(self):
        plan = ResourcePlan.new_default_plan()
        return plan

    def init_job_resource(self, job_resource: JobResource):
        """Adjust the initial resource of typed pods by EasyDL.
        Args:
            job_resource: node resource configuration of a job.
        """
        self._init_job_resource_by_optimizer()
        job_resource.update_node_group_resource(
            NodeType.WORKER,
            self._worker_resource.count,
            self._worker_resource.node_resource.cpu,
            self._worker_resource.node_resource.memory,
        )

        job_resource.update_node_group_resource(
            NodeType.PS,
            self._ps_resource.count,
            self._ps_resource.node_resource.cpu,
            self._ps_resource.node_resource.memory,
        )

        evaluator_group = job_resource.get_node_group_resource(
            NodeType.EVALUATOR
        )
        if evaluator_group:
            resource = evaluator_group.node_resource
            if resource.cpu < NodeResourceLimit.MIN_VALID_CPU:
                resource.cpu = self._worker_resource.node_resource.cpu
            min_memory = NodeResourceLimit.MIN_VALID_MEMORY
            if resource.memory < min_memory:
                resource.memory = self._worker_resource.node_resource.memory

        logger.info("Job resource = %s", job_resource.to_json())
        return job_resource

    def adjust_oom_resource(self, node):
        if node.type == NodeType.PS:
            self._adjust_oom_ps_resource(node)
        else:
            self._adjust_oom_worker_resource(node)

    def _adjust_oom_worker_resource(self, node: Node):
        """Increment the memory to launch worker. The new memory
        is max(1.5 * memory, the memory set by users).

        Args:
            node: Node object.
        """
        cur_mem = node.config_resource.memory
        if (
            _dlrover_context.auto_worker_enabled
            and self._job_stage == JobOptStage.WORKER_INITIAL
        ):
            plan = self._resource_optimizer.generate_oom_recovery_plan(
                [node], JobOptStage.CREATE
            )
            if plan and not plan.empty():
                new_resource = plan.node_group_resources[NodeType.WORKER]
                self._worker_resource.node_resource.memory = max(
                    self._worker_resource.node_resource.memory,
                    new_resource.node_resource.memory,
                )
        else:
            plan = self._get_worker_resource_at_init_phase()
            if NodeType.WORKER in plan.node_group_resources:
                new_resource = self._check_ignore_original_worker_resource(
                    plan.node_group_resources[NodeType.WORKER]
                )
                self._worker_resource.node_resource.memory = max(
                    self._worker_resource.node_resource.memory,
                    new_resource.node_resource.memory,
                )
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        cur_mem = min(cur_mem, NodeResourceLimit.MAX_MEMORY)
        opt_memory = int(
            max(
                self._worker_resource.node_resource.memory,
                cur_mem,
                self._original_worker_resource.node_resource.memory,
            )
        )
        incre_memory = opt_memory - node.config_resource.memory
        incre_memory = min(
            incre_memory, NodeResourceLimit.MAX_INCREMENTAL_MEMORY
        )
        node.config_resource.memory += incre_memory
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )

    def _adjust_oom_ps_resource(self, node: Node):
        """Adjust PS resource if there is a OOM PS"""
        plan = self._resource_optimizer.generate_oom_recovery_plan(
            [node], JobOptStage.PS_INITIAL
        )
        if plan and not plan.empty() and node.name in plan.node_resources:
            resource = plan.node_resources[node.name]
            self._ps_resource.node_resource.memory = max(
                self._ps_resource.node_resource.memory,
                resource.memory,
            )
        cur_mem = node.config_resource.memory
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        opt_memory = int(
            max(
                self._ps_resource.node_resource.memory,
                cur_mem,
                self._original_ps_resource.node_resource.memory,
            )
        )
        incre_memory = opt_memory - node.config_resource.memory
        incre_memory = min(
            incre_memory, NodeResourceLimit.MAX_INCREMENTAL_MEMORY
        )
        node.config_resource.memory += incre_memory
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )
        self._last_ps_change_time = time.time()

    def get_job_resource_plan(self):
        plan = None
        if self._job_stage == JobOptStage.WORKER_INITIAL:
            plan = self._get_worker_resource_at_init_phase()
            self._job_stage = JobOptStage.PS_INITIAL
        elif self._job_stage == JobOptStage.PS_INITIAL:
            plan = self._get_ps_resource_plan()
            self._job_stage = JobOptStage.PS_RUNNING
        elif self._job_stage == JobOptStage.PS_RUNNING:
            plan = self._get_ps_resource_plan()
            if plan.empty():
                plan = self._get_worker_resource_at_running()
        if not plan or plan.empty():
            return None

        if NodeType.WORKER in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.WORKER)

        if plan and NodeType.PS in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.PS)

        plan.adjust_plan_by_context()
        return plan

    def _get_worker_resource_at_running(self):
        if not self.optimize_worker_sampled:
            plan = self._get_worker_resource_at_sample_phase()
            self.optimize_worker_sampled = True
        else:
            plan = self._get_worker_resource_at_stable_phase()
        return plan

    def _get_worker_resource_at_init_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.INITIAL
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if plan.empty():
            logger.info("No any plan to initialize the number of worker")
        return plan

    def _get_worker_resource_at_sample_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.SAMPLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan or plan.empty():
            return
        return plan

    def _get_worker_resource_at_stable_phase(self, optimizer_config={}):
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.STABLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_RUNNING, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_ps_resource_plan(self, optimizer_config={}):
        # The interval of changing PS should be long enough.
        interval = _dlrover_context.seconds_interval_to_change_ps
        if time.time() - self._last_ps_change_time > interval:
            plan = self._resource_optimizer.generate_opt_plan(
                self._job_stage, optimizer_config
            )
        else:
            logger.info(
                "Skip optimizing PS, because the intervalto change ps is too short."
            )
            return ResourcePlan()
        if not plan.empty():
            self._last_ps_change_time = time.time()
        return plan

    def _verify_optimized_group_resource(self, plan: ResourcePlan, node_type):
        group = plan.node_group_resources[node_type]
        if node_type == NodeType.WORKER:
            group = self._check_ignore_original_worker_resource(group)
            node_resource = group.node_resource
            self._worker_resource.count = group.count
            self._worker_resource.node_resource.cpu = node_resource.cpu
            self._worker_resource.node_resource.memory = node_resource.memory
        elif node_type == NodeType.PS:
            group = self._check_ignore_original_ps_resource(group)
            node_resource = group.node_resource
            self._ps_resource.count = min(
                group.count, NodeResourceLimit.MAX_PS_NUM
            )
            self._ps_resource.node_resource.cpu = node_resource.cpu
            self._ps_resource.node_resource.memory = node_resource.memory
        return group

    def _check_ignore_original_worker_resource(
        self, resource: NodeGroupResource
    ):
        """Abandon the optimization result if users have set the resource."""
        #  Users may worry about that the increasing number of worker hurts the
        #  accuracy, so the max number of worker is the configuration.
        original_resource = self._original_worker_resource.node_resource
        if self._original_worker_resource.count > 0:
            resource.count = self._original_worker_resource.count
        if resource.node_resource.cpu == 0:
            resource.node_resource.cpu = original_resource.cpu
        if resource.node_resource.memory == 0:
            resource.node_resource.memory = original_resource.memory
        return resource

    def _check_ignore_original_ps_resource(self, resource: NodeGroupResource):
        """Abandon the optimization result if users have set the resource."""
        original_resource = self._original_ps_resource.node_resource
        if self._original_ps_resource.count > 0:
            resource.count = self._original_ps_resource.count
        if original_resource.memory >= NodeResourceLimit.MIN_VALID_MEMORY:
            resource.node_resource.memory = original_resource.memory
        if original_resource.cpu >= NodeResourceLimit.MIN_VALID_CPU:
            resource.node_resource.cpu = original_resource.cpu
        return resource


class AllreduceJobResourceOptimizer(JobResourceOptimizer):
    """It generates resource configuration for a job."""

    def __init__(
        self,
        worker_resource: NodeGroupResource,
        job_uuid="",
    ):
        self._worker_resource = worker_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._job_uuid = job_uuid
        self._lock = threading.Lock()
        self._node_unit = 1
        self._alive_node_num = 0

    def update_job_uuid(self, job_uuid):
        pass

    def init_job_resource(self, job_resource: JobResource):
        pass

    def get_job_resource_plan(self) -> ResourcePlan:
        """Check whether there are free nodes in the cluster."""
        plan = ResourcePlan()
        worker_config = copy.deepcopy(self._original_worker_resource)
        max_node_num = self._original_worker_resource.count
        request_num = max_node_num - self._alive_node_num
        free_num = self._get_free_gpu_node()
        free_num = (free_num // self._node_unit) * self._node_unit
        new_num = min(free_num, request_num)
        worker_config.count = self._alive_node_num + new_num
        plan.node_group_resources[NodeType.WORKER] = worker_config
        return plan

    # TODO: implement the function to query the number free GPU nodes.
    def _get_free_gpu_node(self):
        return 0

    def adjust_oom_resource(self, node: Node):
        """Adjust the resource configuration for OOM nodes"""
        # no adjustment for now(for allreduce type)
        pass

    def get_config_resource(self):
        job_config = JobResource()
        worker_config = self._original_worker_resource
        job_config.node_group_resources[NodeType.WORKER] = worker_config
        return job_config

    def set_node_unit(self, node_unit):
        self._node_unit = node_unit

    def set_alive_node_num(self, node_num):
        self._alive_node_num = node_num
