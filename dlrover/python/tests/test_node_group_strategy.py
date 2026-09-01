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
    _resolve_stripe_group_id,
    resolve_group_id,
    validate_topology,
)


def _ep_pp_dp(tp=1, pp=1, ep=1, cp=1, num_nodes=0, ranks_per_node=0):
    return NodeGroupSchedule(
        strategy=NodeGroupStrategy.EP_PP_DP,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        num_nodes=num_nodes,
        ranks_per_node=ranks_per_node,
    )


class NodeGroupScheduleTest(unittest.TestCase):
    def test_defaults_are_contiguous_unsized(self):
        sched = NodeGroupSchedule()
        self.assertEqual(sched.strategy, NodeGroupStrategy.CONTIGUOUS)
        self.assertEqual(
            (sched.tp, sched.pp, sched.ep, sched.cp), (1, 1, 1, 1)
        )
        self.assertEqual((sched.num_nodes, sched.ranks_per_node), (0, 0))

    def test_ep_pp_dp_carry_fields(self):
        sched = _ep_pp_dp(
            tp=1, pp=16, ep=32, cp=1, num_nodes=1152, ranks_per_node=8
        )
        self.assertEqual(sched.strategy, NodeGroupStrategy.EP_PP_DP)
        self.assertEqual((sched.pp, sched.ep), (16, 32))
        self.assertEqual((sched.num_nodes, sched.ranks_per_node), (1152, 8))


class ResolveGroupIdStripeTest(unittest.TestCase):
    def test_no_affinity_or_non_worker_returns_none(self):
        self.assertIsNone(resolve_group_id(None, NodeType.WORKER, 0, None))
        self.assertIsNone(resolve_group_id({}, NodeType.WORKER, 0, None))
        self.assertIsNone(resolve_group_id({0: 2}, NodeType.PS, 0, None))

    def test_schedule_none_falls_back_to_contiguous(self):
        ga = {0: 2, 1: 2}
        self.assertEqual(
            [resolve_group_id(ga, NodeType.WORKER, k, None) for k in range(4)],
            [0, 0, 1, 1],
        )

    def test_contiguous_explicit_schedule(self):
        ga = {0: 2, 1: 2}
        sched = NodeGroupSchedule(strategy=NodeGroupStrategy.CONTIGUOUS)
        self.assertEqual(
            [
                resolve_group_id(ga, NodeType.WORKER, k, sched)
                for k in range(4)
            ],
            [0, 0, 1, 1],
        )

    def test_stripe_formula_small(self):
        # R=8, EP=8, PP=2, G=2, TP=CP=1: DP_nodes = N/(1*2*1) = 2; seg = 1.
        # group(k) = (k % 2) // 1 = k % 2.
        ga = {0: 2, 1: 2}
        sched = _ep_pp_dp(
            tp=1, pp=2, ep=8, cp=1, num_nodes=4, ranks_per_node=8
        )
        self.assertEqual(
            [
                resolve_group_id(ga, NodeType.WORKER, k, sched)
                for k in range(4)
            ],
            [0, 1, 0, 1],
        )

    def test_stripe_out_of_range_rank_raises(self):
        # A validated ep_pp_dp topology is fixed-size: a rank beyond
        # num_nodes (e.g. an unexpected scale-up) must fail loudly
        # instead of wrapping back into an already-full segment.
        ga = {0: 2, 1: 2}
        sched = _ep_pp_dp(
            tp=1, pp=2, ep=8, cp=1, num_nodes=4, ranks_per_node=8
        )
        with self.assertRaises(AssertionError):
            resolve_group_id(ga, NodeType.WORKER, 4, sched)
        with self.assertRaises(AssertionError):
            resolve_group_id(ga, NodeType.WORKER, -1, sched)
        # The last in-range rank still resolves normally.
        self.assertEqual(resolve_group_id(ga, NodeType.WORKER, 3, sched), 1)

    def test_stripe_matches_formula_helper_on_9216_job(self):
        # N=1152, R=8, TP=1, PP=16, EP=32, CP=1, G=9.
        N, R, PP, EP, G = 1152, 8, 16, 32, 9
        ga = {i: N // G for i in range(G)}
        sched = _ep_pp_dp(
            tp=1, pp=PP, ep=EP, cp=1, num_nodes=N, ranks_per_node=R
        )
        dp_nodes = N // (1 * PP * 1)
        seg = dp_nodes // G
        for k in range(N):
            self.assertEqual(
                resolve_group_id(ga, NodeType.WORKER, k, sched),
                _resolve_stripe_group_id(ga, k, sched),
            )
            self.assertEqual(
                resolve_group_id(ga, NodeType.WORKER, k, sched),
                (k % dp_nodes) // seg,
            )
        # boundary checks: first 8 nodes -> seg0; node 8 -> seg1; node 71 -> seg8;
        # node 72 (head of the next pipeline stage) -> seg0 again.
        self.assertEqual(
            [
                resolve_group_id(ga, NodeType.WORKER, k, sched)
                for k in range(8)
            ],
            [0] * 8,
        )
        self.assertEqual(resolve_group_id(ga, NodeType.WORKER, 8, sched), 1)
        self.assertEqual(resolve_group_id(ga, NodeType.WORKER, 71, sched), 8)
        self.assertEqual(resolve_group_id(ga, NodeType.WORKER, 72, sched), 0)


class ValidateTopologyTest(unittest.TestCase):
    def test_ok_9216_job(self):
        ga = {i: 128 for i in range(9)}
        validate_topology(
            ga,
            _ep_pp_dp(
                tp=1, pp=16, ep=32, cp=1, num_nodes=1152, ranks_per_node=8
            ),
        )

    def test_ok_small(self):
        ga = {0: 2, 1: 2}
        validate_topology(
            ga,
            _ep_pp_dp(tp=1, pp=2, ep=8, cp=1, num_nodes=4, ranks_per_node=8),
        )

    def test_contiguous_or_none_is_noop(self):
        # A deliberately unequal/invalid-for-ep_pp_dp mapping is NOT checked at
        # all when the strategy is contiguous or the schedule is None.
        ga = {0: 10, 1: 5}
        validate_topology(
            ga, NodeGroupSchedule(strategy=NodeGroupStrategy.CONTIGUOUS)
        )
        validate_topology(ga, None)

    def _assert_raises_msg(self, ga, sched, needle):
        with self.assertRaises(ValueError) as ctx:
            validate_topology(ga, sched)
        self.assertIn(needle, str(ctx.exception))

    def test_requires_group_affinity(self):
        s = _ep_pp_dp(num_nodes=10, ranks_per_node=8)
        self._assert_raises_msg(None, s, "requires --group-affinity")
        self._assert_raises_msg({}, s, "requires --group-affinity")

    def test_tp_must_be_one(self):
        ga = {0: 128, 1: 128, 2: 128}
        sched = _ep_pp_dp(
            tp=2, pp=4, ep=8, cp=1, num_nodes=384, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "requires TP=1")

    def test_cp_must_be_one(self):
        ga = {0: 128, 1: 128, 2: 128}
        sched = _ep_pp_dp(
            tp=1, pp=4, ep=8, cp=2, num_nodes=384, ranks_per_node=8
        )
        with self.assertRaises(ValueError):
            validate_topology(ga, sched)

    def test_ranks_per_node_positive(self):
        ga = {0: 64, 1: 64}
        sched = _ep_pp_dp(
            tp=1, pp=2, ep=8, cp=1, num_nodes=128, ranks_per_node=0
        )
        self._assert_raises_msg(ga, sched, "ranks_per_node>0")

    def test_dense_dp_must_divide_G(self):
        # N=1152, R=8, PP=16 -> dense_dp=576; G=5 -> 576%5 != 0 (constraint A).
        ga = {
            i: 230 for i in range(5)
        }  # sizes irrelevant: A is checked before D
        sched = _ep_pp_dp(
            tp=1, pp=16, ep=32, cp=1, num_nodes=1152, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "divisible by G=5")

    def test_segment_block_not_node_aligned(self):
        # N=3, R=8, PP=1, G=2: dense_dp=24, S=12; 12%8 != 0 (constraint B).
        ga = {0: 1, 1: 1}
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=8, cp=1, num_nodes=3, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "divisible by ranks_per_node R=8")

    def test_ep_too_large_for_segment(self):
        # N=4, R=8, PP=1, G=2, EP=24: dense_dp=32, S=16; 16%24 != 0 (constraint C).
        ga = {0: 2, 1: 2}
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=24, cp=1, num_nodes=4, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "EP size (24) to")

    def test_ep_not_whole_nodes(self):
        # N=12, R=8, PP=1, G=2, EP=12: dense_dp=96, S=48; S%EP=0 (C ok),
        # S%R=0 (B ok), but EP%R=12%8 != 0 (constraint E).
        ga = {0: 6, 1: 6}
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=12, cp=1, num_nodes=12, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "divisible by ranks_per_node R=8")

    def test_unequal_group_sizes_fail(self):
        # N=12, R=8, PP=1, G=2, EP=8: dense_dp=96, S=48; A/B/C/E all pass,
        # N%G=0, N/G=6. Unequal sizes -> constraint D fails.
        ga = {0: 7, 1: 5}
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=8, cp=1, num_nodes=12, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "equal")

    def test_wrong_equal_size_fail(self):
        # Equal sizes but not N/G.
        ga = {0: 5, 1: 5}
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=8, cp=1, num_nodes=12, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "N/G=6")

    def test_non_contiguous_group_ids_fail(self):
        ga = {0: 4, 1: 4, 3: 4}  # missing group id 2
        sched = _ep_pp_dp(
            tp=1, pp=1, ep=8, cp=1, num_nodes=12, ranks_per_node=8
        )
        self._assert_raises_msg(ga, sched, "contiguous group ids")


class StripeInvariantsTest(unittest.TestCase):
    """Rank-level invariants for a validated topology: EP groups and PP
    columns stay inside one segment; each DP group spans every segment."""

    def _check_invariants(self, N, R, TP, PP, EP, CP, G):
        ga = {i: N // G for i in range(G)}
        sched = _ep_pp_dp(
            tp=TP, pp=PP, ep=EP, cp=CP, num_nodes=N, ranks_per_node=R
        )
        validate_topology(ga, sched)

        def seg_of_rank(rank):
            return resolve_group_id(ga, NodeType.WORKER, rank // R, sched)

        dense_dp = (N * R) // (TP * PP * CP)

        # EP group: fixed (pp, de), vary ep over [0, EP) -> one segment.
        for pp in range(PP):
            for de in range(dense_dp // EP):
                segs = set(
                    seg_of_rank(pp * dense_dp + de * EP + e) for e in range(EP)
                )
                self.assertEqual(len(segs), 1, f"EP group pp={pp} de={de}")

        # PP column: fixed dp, vary pp over [0, PP) -> one segment.
        for dp in range(dense_dp):
            segs = set(seg_of_rank(dp + pp * dense_dp) for pp in range(PP))
            self.assertEqual(len(segs), 1, f"PP column dp={dp}")

        # DP group: fixed pp, vary dp over [0, dense_dp) -> all G segments.
        for pp in range(PP):
            segs = set(
                seg_of_rank(pp * dense_dp + dp) for dp in range(dense_dp)
            )
            self.assertEqual(segs, set(range(G)), f"DP group pp={pp}")

    def test_9216_job(self):
        self._check_invariants(N=1152, R=8, TP=1, PP=16, EP=32, CP=1, G=9)

    def test_small_compact(self):
        # 1 node / EP group / segment per stage.
        self._check_invariants(N=4, R=8, TP=1, PP=2, EP=8, CP=1, G=2)

    def test_two_segments_more_stages(self):
        # DP_nodes=2, seg=1 node per segment per stage.
        self._check_invariants(N=8, R=8, TP=1, PP=4, EP=8, CP=1, G=2)


if __name__ == "__main__":
    unittest.main()
