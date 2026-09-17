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

import unittest

from dlrover.python.common.constants import NodeGroupStrategy, NodeType
from dlrover.python.master.resource.job import (
    NodeGroupSchedule,
    resolve_group_id,
    validate_topology,
)
from dlrover.python.master.resource.soft_group import (
    SoftGroupSchedule,
    resolve_soft_group_id,
    validate_soft_group_topology,
)


def _soft(
    strategy=NodeGroupStrategy.EP_DP_PP,
    sizes=None,
    tp=1,
    pp=1,
    ep=1,
    etp=None,
    cp=1,
    num_nodes=None,
    ranks_per_node=8,
    no_group_failover=False,
):
    if sizes is None:
        sizes = {0: 1}
    return SoftGroupSchedule(
        strategy=strategy,
        sizes=sizes,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        cp=cp,
        num_nodes=(
            num_nodes if num_nodes is not None else sum(sizes.values())
        ),
        ranks_per_node=ranks_per_node,
        no_group_failover=no_group_failover,
    )


class SoftGroupScheduleTest(unittest.TestCase):
    def test_carry_fields(self):
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 30, 1: 20},
            pp=2,
            ep=40,
            etp=2,
            no_group_failover=True,
        )
        self.assertEqual(schedule.sizes, {0: 30, 1: 20})
        self.assertEqual((schedule.pp, schedule.ep, schedule.etp), (2, 40, 2))
        self.assertEqual((schedule.tp, schedule.cp), (1, 1))
        self.assertEqual(
            (schedule.num_nodes, schedule.ranks_per_node), (50, 8)
        )
        self.assertTrue(schedule.no_group_failover)
        self.assertIsNone(schedule._ep_pp_dp_layout)

    def test_unset_etp_inherits_tp(self):
        # An unset etp (None) inherits the recorded tp; an explicit etp
        # wins over tp.
        self.assertEqual(_soft(tp=4).etp, 4)
        self.assertEqual(_soft(tp=4, etp=2).etp, 2)


class ValidateSoftGroupTopologyTest(unittest.TestCase):
    def test_none_or_empty_is_noop(self):
        validate_soft_group_topology(None)
        validate_soft_group_topology(_soft(sizes={}))

    def test_ok_unequal(self):
        # The whole point of the soft path: heterogeneous sizes are legal
        # (EP slot = 1 pod with ep=8, R=8, so any size is aligned).
        validate_soft_group_topology(_soft(sizes={0: 30, 1: 20}, pp=2, ep=8))

    def test_ok_ep_slot_aligned(self):
        # N=64, R=8, EP=16, PP=2 -> dense_dp=256, dp=16, ep_workers=2;
        # every size is a multiple of 2. EP alignment is the default, no
        # flag involved.
        validate_soft_group_topology(
            _soft(
                strategy=NodeGroupStrategy.EP_PP_DP,
                sizes={0: 32, 1: 32},
                pp=2,
                ep=16,
            )
        )
        # {0:30, 1:20} with ep=40: dp=5 and ep_workers=5 divides both sizes.
        validate_soft_group_topology(
            _soft(
                sizes={0: 30, 1: 20},
                pp=2,
                ep=40,
            )
        )

    def test_sum_must_equal_num_nodes(self):
        with self.assertRaisesRegex(ValueError, "sum to 50"):
            validate_soft_group_topology(
                _soft(sizes={0: 30, 1: 20}, num_nodes=49, pp=2, ep=8)
            )

    def test_positive_sizes_and_keys(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_soft_group_topology(_soft(sizes={0: 0, 1: 4}, pp=1, ep=1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_soft_group_topology(
                _soft(sizes={-1: 4, 0: 4}, pp=1, ep=1)
            )

    def test_tp_cp_recorded_not_used(self):
        # TP and CP are recorded but not validated: tp>1 / cp>1 used to
        # fail the check with "requires TP=1"/"requires CP=1"; they now
        # pass through untouched (and an unset etp inherits tp).
        schedule = _soft(sizes={0: 8, 1: 8}, tp=2, cp=2, pp=2, ep=8)
        validate_soft_group_topology(schedule)
        self.assertEqual((schedule.tp, schedule.etp, schedule.cp), (2, 2, 2))

    def test_invalid_parallel_sizes(self):
        with self.assertRaisesRegex(ValueError, "pp=0"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=0, ep=8))
        with self.assertRaisesRegex(ValueError, "ep=0"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=2, ep=0))
        with self.assertRaisesRegex(ValueError, "ranks_per_node"):
            validate_soft_group_topology(
                _soft(sizes={0: 8, 1: 8}, pp=2, ep=8, ranks_per_node=0)
            )

    def test_num_nodes_must_divide_model_parallel(self):
        with self.assertRaisesRegex(ValueError, r"PP=3"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=3, ep=8))

    def test_dp_must_be_integral(self):
        # N=50, R=8, PP=2 -> dense_dp=200 is not divisible by the EP
        # group size EPG=16: not even a valid Megatron shape (dp=12.5).
        with self.assertRaisesRegex(ValueError, "EP group size"):
            validate_soft_group_topology(
                _soft(sizes={0: 30, 1: 20}, pp=2, ep=16)
            )

    def test_ep_pp_dp_requires_ep_multiple_of_r(self):
        # N=24, R=8, PP=2 -> dense_dp=96 divides EPG=12 (dp integral) but
        # EPG=12 is not a multiple of R=8, so an EP slot is not whole
        # nodes.
        with self.assertRaisesRegex(
            ValueError, "ETP\\*EP=12 to be divisible by ranks_per_node"
        ):
            validate_soft_group_topology(
                _soft(
                    strategy=NodeGroupStrategy.EP_PP_DP,
                    sizes={0: 12, 1: 12},
                    pp=2,
                    ep=12,
                )
            )

    def test_alignment_is_default_not_optin(self):
        # dense_dp=256 with pp=2, EP=16 -> dp integral and EP%R==0, but
        # 31 is odd so the group is not a multiple of the EP slot
        # (ep_workers=2). No flag exists to relax this: the master fails
        # to start for BOTH strategies.
        for strategy in (
            NodeGroupStrategy.EP_DP_PP,
            NodeGroupStrategy.EP_PP_DP,
        ):
            with self.assertRaisesRegex(ValueError, "group 0 has 31"):
                validate_soft_group_topology(
                    _soft(
                        strategy=strategy,
                        sizes={0: 31, 1: 33},
                        pp=2,
                        ep=16,
                    )
                )

    def test_alignment_requires_ep_multiple_of_r_even_contiguous(self):
        # The EP slot must be well-defined regardless of the strategy:
        # N=48, R=8, PP=2 -> dense_dp=192 divides EPG=12, but EPG=12 is
        # not a multiple of R=8.
        with self.assertRaisesRegex(ValueError, "ETP\\*EP=12 to be"):
            validate_soft_group_topology(
                _soft(
                    sizes={0: 33, 1: 15},
                    pp=2,
                    ep=12,
                )
            )


class ResolveSoftGroupIdTest(unittest.TestCase):
    def test_none_or_non_worker_returns_none(self):
        self.assertIsNone(resolve_soft_group_id(None, NodeType.WORKER, 0))
        self.assertIsNone(resolve_soft_group_id(_soft(sizes={}), "worker", 0))
        schedule = _soft(sizes={0: 4, 1: 4})
        self.assertIsNone(resolve_soft_group_id(schedule, "ps", 0))

    def test_contiguous_unequal_sizes(self):
        schedule = _soft(sizes={0: 30, 1: 20})
        result = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in (0, 29, 30, 49, 50)
        ]
        self.assertEqual(result, [0, 0, 1, 1, None])

    def test_contiguous_uses_ascending_group_keys(self):
        schedule = _soft(sizes={5: 2, 2: 3})
        result = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in range(5)
        ]
        self.assertEqual(result, [2, 2, 2, 5, 5])

    def test_ep_pp_dp_equal_sizes_slot_round_robin(self):
        # N=16, R=8, PP=2, EP=16 -> dp_nodes=8, ep_workers=2: one EP slot
        # (2 pods) from every group per round.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 8, 1: 8},
            pp=2,
            ep=16,
        )
        self.assertEqual(
            [
                resolve_soft_group_id(schedule, NodeType.WORKER, rank)
                for rank in range(16)
            ],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_ep_pp_dp_etp_enlarges_ep_slot(self):
        # etp=2 with ep=8, R=8 -> the EP slot is ETP*EP/R = 2 pods, i.e.
        # the SAME layout as ep=16 without etp: every 2-pod slot stays in
        # one group across both pipeline stages.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 8, 1: 8},
            pp=2,
            ep=8,
            etp=2,
        )
        validate_soft_group_topology(schedule)
        self.assertEqual(
            [
                resolve_soft_group_id(schedule, NodeType.WORKER, rank)
                for rank in range(16)
            ],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_ep_pp_dp_leftover_pods_tolerated_with_warning(self):
        # Defensive resolver behavior for a schedule that bypassed the
        # (now mandatory) EP alignment validation: {0:9, 1:7} has 1+1
        # leftover pods after the whole slots; they complete the layout
        # pod-by-pod, so the final EP slot straddles both groups —
        # reported with a WARNING, never a resolver failure.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 9, 1: 7},
            pp=2,
            ep=16,
        )
        with self.assertLogs("dlrover.logger", level="WARNING") as logs:
            layout = [
                resolve_soft_group_id(schedule, NodeType.WORKER, rank)
                for rank in range(16)
            ]
        self.assertEqual(
            layout,
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        )
        self.assertTrue(any("crossing groups" in log for log in logs.output))

    def test_ep_pp_dp_unequal_sizes_per_stage_rotation(self):
        # The ep_pp_dp rotation RESETS at every pipeline stage (virtual
        # group) boundary instead of running globally: with sizes
        # {1:4, 2:2, 3:2}, pp=2, ep_workers=2 the layout matches the
        # requested segment placement — SG1 keeps its PP column
        # intra-group (pods 0,1 <-> 4,5), and ONLY the odd-slot groups
        # SG2/SG3 pair across stages (unavoidable with 1 slot each).
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={1: 4, 2: 2, 3: 2},
            pp=2,
            ep=16,
        )
        validate_soft_group_topology(schedule)
        layout = [
            resolve_soft_group_id(schedule, NodeType.WORKER, k)
            for k in range(8)
        ]
        # stage 0 = [1,1, 2,2], stage 1 = [1,1, 3,3]
        self.assertEqual(layout, [1, 1, 2, 2, 1, 1, 3, 3])
        self.assertEqual([k for k in range(8) if layout[k] == 1], [0, 1, 4, 5])
        self.assertEqual([k for k in range(8) if layout[k] == 2], [2, 3])
        self.assertEqual([k for k in range(8) if layout[k] == 3], [6, 7])
        # PP column alignment: same within-stage position maps to the
        # same group except the odd-slot SG2/SG3 pair.
        dp_nodes, pp = 4, 2
        straddled = [
            p
            for p in range(dp_nodes)
            if layout[p] != layout[pp * dp_nodes // 2 + p]
        ]
        self.assertEqual(straddled, [2, 3])
        # All EP slots stay intra-group.
        for start in range(0, 8, 2):
            self.assertEqual(len(set(layout[start : start + 2])), 1)

    def test_ep_pp_dp_greedy_column_packing_never_starves(self):
        # {g1: 6, g2: 2, g3: 2} with 1-pod EP slots starves a per-stage
        # round robin (the small groups drain in stage 0 and stage 1
        # cannot be filled); the greedy column packing lays the world
        # out fully: each pair of same-position slots forms a column
        # owned by ONE group, so every PP column stays intra-group.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={1: 6, 2: 2, 3: 2},
            pp=2,
            ep=8,
        )
        validate_soft_group_topology(schedule)
        layout = [
            resolve_soft_group_id(schedule, NodeType.WORKER, k)
            for k in range(10)
        ]
        self.assertEqual(layout, [1, 1, 1, 2, 3, 1, 1, 1, 2, 3])
        # columns: [1,1],[1,1],[1,1],[2,2],[3,3] -> column s-th slots
        dp_nodes, pp = 5, 2
        for position in range(dp_nodes):
            self.assertEqual(
                layout[position], layout[pp * 0 + dp_nodes + position]
            )

    def _check_scenario(
        self,
        sizes,
        pp,
        ep,
        expected_layout,
        mixed_columns,
        ranks_per_node=8,
    ):
        """Assert a full ep_pp_dp scenario: the greedy column packing
        produces exactly ``expected_layout`` (verified against the real
        implementation), every group hosts its declared pod count, every
        EP slot stays inside ONE group, and the number of PP columns
        mixing two groups matches the theoretical minimum
        (``ceil(odd_slot_groups / 2)`` plateaus) given by
        ``mixed_columns`` (None skips the mixed-count assertion)."""
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes=sizes,
            pp=pp,
            ep=ep,
            ranks_per_node=ranks_per_node,
        )
        validate_soft_group_topology(schedule)
        total = sum(sizes.values())
        layout = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in range(total)
        ]
        self.assertEqual(layout, expected_layout)
        # every group keeps exactly its declared pod count
        for group_id, size in sizes.items():
            self.assertEqual(layout.count(group_id), size)
        slot = ep // ranks_per_node
        dp_nodes = total // pp
        ep_groups = dp_nodes // slot
        # every EP slot is single-group (never straddles a segment)
        for stage in range(pp):
            for de in range(ep_groups):
                pods = [stage * dp_nodes + de * slot + i for i in range(slot)]
                self.assertEqual(
                    len({layout[p] for p in pods}),
                    1,
                    f"EP(s={stage}, de={de}) pods {pods} straddle",
                )
        if mixed_columns is None:
            return
        # compute the theoretical minimum of mixed PP columns
        odd_slot_groups = sum(
            1 for size in sizes.values() if (size // slot) % pp != 0
        )
        self.assertLessEqual(
            mixed_columns,
            (odd_slot_groups + 1) // 2,
            "greedy column packing pays more mixed columns than the "
            "pairing lower bound",
        )
        actual_mixed = 0
        for de in range(ep_groups):
            if any(
                layout[stage * dp_nodes + de * slot]
                != layout[(stage + 1) * dp_nodes + de * slot]
                for stage in range(pp - 1)
            ):
                actual_mixed += 1
        self.assertEqual(actual_mixed, mixed_columns)

    def test_ep_pp_dp_scenario_matrix_legal(self):
        # A spectrum of legal topologies: single group, equal sizes,
        # unequal multiples of 4, pp=4, four segments, 1-pod slots...
        # (expected layouts verified against the real implementation).
        self._check_scenario(
            sizes={7: 8},
            pp=2,
            ep=16,
            expected_layout=[7] * 8,
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 2, 2: 2},
            pp=2,
            ep=8,
            expected_layout=[1, 2, 1, 2],
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 8, 2: 8},
            pp=2,
            ep=16,
            expected_layout=[1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 8, 2: 4},
            pp=2,
            ep=16,
            expected_layout=[1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 8, 2: 4, 3: 4},
            pp=4,
            ep=8,
            expected_layout=[1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3],
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 2, 2: 2, 3: 2, 4: 2},
            pp=2,
            ep=8,
            expected_layout=[1, 2, 3, 4, 1, 2, 3, 4],
            mixed_columns=0,
        )
        self._check_scenario(
            sizes={1: 12, 2: 8, 3: 8},
            pp=2,
            ep=16,
            expected_layout=[
                1,
                1,
                1,
                1,
                2,
                2,
                3,
                3,
                1,
                1,
                2,
                2,
                3,
                3,
                1,
                1,
                1,
                1,
                2,
                2,
                3,
                3,
                1,
                1,
                2,
                2,
                3,
                3,
            ],
            mixed_columns=0,
        )
        # two odd-slot groups: the two residual EP slots pair in ONE
        # mixed column (the theoretical minimum).
        self._check_scenario(
            sizes={1: 6, 2: 6, 3: 4},
            pp=2,
            ep=16,
            expected_layout=[1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2],
            mixed_columns=1,
        )
        # the user-provided S2/S3 shapes
        self._check_scenario(
            sizes={1: 4, 2: 2, 3: 2},
            pp=2,
            ep=16,
            expected_layout=[1, 1, 2, 2, 1, 1, 3, 3],
            mixed_columns=1,
        )
        self._check_scenario(
            sizes={1: 6, 2: 4, 3: 2},
            pp=2,
            ep=16,
            expected_layout=[1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 3, 3],
            mixed_columns=1,
        )
        # the fragment-pairing minimum with 1-pod EP slots: the three
        # whole columns and the two 1-slot residual fragments pair up
        # in the last column — ONE mixed column (a descending chained
        # fill pays two).
        self._check_scenario(
            sizes={1: 3, 2: 3, 3: 2},
            pp=2,
            ep=8,
            expected_layout=[1, 2, 3, 1, 1, 2, 3, 2],
            mixed_columns=1,
        )
        # the starvation counter-example with 1-pod slots: a per-stage
        # round robin drains the small groups in stage 0; the greedy
        # column packing lays the world out fully.
        self._check_scenario(
            sizes={1: 6, 2: 2, 3: 2},
            pp=2,
            ep=8,
            expected_layout=[1, 1, 1, 2, 3, 1, 1, 1, 2, 3],
            mixed_columns=0,
        )

    def test_ep_pp_dp_scenario_matrix_invalid(self):
        # A spectrum of illegal topologies rejected by the start-up
        # validation, each failing fast with its own message.
        invalid_cases = [
            # dense_dp = 10*8/2 = 40 not divisible by EPG=EP=16 (N must
            # be a multiple of 4 for pp=2, ep=16, R=8)
            ({1: 6, 2: 2, 3: 2}, 2, 16, "EP group size"),
            # group 1 has 3 pods, not a multiple of the EP slot (2)
            ({1: 3, 2: 3, 3: 2}, 2, 16, "ep_workers=2"),
            # N=6 -> dense_dp=24 not divisible by EPG=EP=16
            ({1: 4, 2: 2}, 2, 16, "EP group size"),
        ]
        for sizes, pp, ep, pattern in invalid_cases:
            schedule = _soft(
                strategy=NodeGroupStrategy.EP_PP_DP,
                sizes=sizes,
                pp=pp,
                ep=ep,
            )
            with self.assertRaisesRegex(ValueError, pattern):
                validate_soft_group_topology(schedule)
        # EPG=EP=12 passes dense_dp divisibility (48 % 12 == 0) but is
        # not a multiple of R=8: an EP slot would not consist of whole
        # nodes
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={1: 6, 2: 6},
            pp=2,
            ep=12,
        )
        with self.assertRaisesRegex(
            ValueError, "ETP\\*EP=12 to be divisible by ranks_per_node"
        ):
            validate_soft_group_topology(schedule)
        # declared N mismatches the sum of sizes
        schedule = _soft(sizes={1: 4, 2: 4}, pp=2, ep=16, num_nodes=7)
        with self.assertRaisesRegex(ValueError, "sum to 8"):
            validate_soft_group_topology(schedule)

    def test_ep_pp_dp_layout_built_once(self):
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 8, 1: 8},
            pp=2,
            ep=16,
        )
        first = resolve_soft_group_id(schedule, NodeType.WORKER, 3)
        layout = schedule._ep_pp_dp_layout
        second = resolve_soft_group_id(schedule, NodeType.WORKER, 3)
        self.assertEqual(first, second)
        self.assertIs(layout, schedule._ep_pp_dp_layout)

    def test_ep_pp_dp_equal_sizes_matches_stripe_formula(self):
        # When every group holds exactly one EP slot per pipeline stage
        # (size = PP*EP/R, i.e. N*R = PP*EP*G) the round-robin layout
        # reproduces the static ep_pp_dp stripe formula exactly, for
        # contiguous AND opaque (non 0-based, per #1745) group ids.
        N, R, PP, EP, G = 16, 8, 2, 16, 4
        for ga in (
            {i: N // G for i in range(G)},  # contiguous ids {0,1,2,3}
            {5: 4, 9: 4, 17: 4, 23: 4},  # opaque ids
        ):
            static = NodeGroupSchedule(
                strategy=NodeGroupStrategy.EP_PP_DP,
                tp=1,
                pp=PP,
                ep=EP,
                cp=1,
                num_nodes=N,
                ranks_per_node=R,
            )
            validate_topology(ga, static)
            schedule = _soft(
                strategy=NodeGroupStrategy.EP_PP_DP,
                sizes=ga,
                pp=PP,
                ep=EP,
            )
            self.assertEqual(
                [
                    resolve_soft_group_id(schedule, NodeType.WORKER, k)
                    for k in range(N)
                ],
                [
                    resolve_group_id(ga, NodeType.WORKER, k, static)
                    for k in range(N)
                ],
            )

    def test_ep_pp_dp_invariants_9216_equal_sizes(self):
        # N=1152 = 9 segments x 128 workers, R=8, PP=16, EP=32:
        # ep_workers=4, DP_nodes=72 (2 rounds over the 9 groups per
        # stage). EP slots stay inside one group and the same
        # within-stage position always maps to the same group.
        N, R, PP, EP, G = 1152, 8, 16, 32, 9
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={g: N // G for g in range(G)},
            pp=PP,
            ep=EP,
        )
        layout = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in range(N)
        ]
        dp_nodes, slot = N // PP, EP // R
        for start in range(0, N, slot):
            self.assertEqual(
                len(set(layout[start : start + slot])),
                1,
                f"EP slot at [{start},{start + slot})",
            )
        for position in range(dp_nodes):
            self.assertEqual(
                len({layout[pp * dp_nodes + position] for pp in range(PP)}),
                1,
                f"PP column {position}",
            )
        sums = {g: layout.count(g) for g in range(G)}
        self.assertEqual(sums, {g: N // G for g in range(G)})


if __name__ == "__main__":
    unittest.main()
