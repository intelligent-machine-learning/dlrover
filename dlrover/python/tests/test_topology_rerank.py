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
import tempfile
import unittest
from typing import Dict
from unittest.mock import patch

from dlrover.python.common.constants import NodeGroupStrategy
from dlrover.python.master.elastic_training.net_topology import (
    NodeTopologyMeta,
)
from dlrover.python.master.elastic_training.topology_rerank import (
    GROUP_TOPOLOGY_ENV,
    GROUP_TOPOLOGY_FILE_ENV,
    GroupTopologyQuerier,
    GroupTopologySorter,
    _ep_dp_pp_group_sequence,
    _ep_pp_dp_group_sequence,
)


def _meta(node_rank, psw, process_num=8):
    return NodeTopologyMeta(
        node_id=node_rank,
        node_rank=node_rank,
        process_num=process_num,
        node_ip=f"10.0.0.{node_rank}",
        asw="",
        psw=psw,
    )


def _nodes(
    psw_by_rank: Dict[int, str], process_num: int = 8
) -> Dict[int, NodeTopologyMeta]:
    return {
        rank: _meta(rank, psw, process_num=process_num)
        for rank, psw in psw_by_rank.items()
    }


class GroupTopologyQuerierTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(GROUP_TOPOLOGY_ENV, None)
        os.environ.pop(GROUP_TOPOLOGY_FILE_ENV, None)

    def tearDown(self):
        os.environ.pop(GROUP_TOPOLOGY_ENV, None)
        os.environ.pop(GROUP_TOPOLOGY_FILE_ENV, None)

    def test_query_from_env(self):
        topology = {"10.0.0.1": "SG2", "10.0.0.2": "SG10"}
        with patch.dict(
            os.environ, {GROUP_TOPOLOGY_ENV: json.dumps(topology)}
        ):
            querier = GroupTopologyQuerier()
        self.assertEqual(querier.query("10.0.0.1"), ("", "SG2"))
        self.assertEqual(querier.query("10.0.0.2"), ("", "SG10"))
        # asw stays empty (one-tier topology) and unknown IPs fall into
        # the unknown group.
        self.assertEqual(querier.query("10.0.0.3"), ("", ""))

    def test_query_from_file(self):
        topology = {"10.0.0.1": "SG1", "10.0.0.2": "SG1"}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as topo_file:
            json.dump(topology, topo_file)
            topo_file.flush()
            with patch.dict(
                os.environ, {GROUP_TOPOLOGY_FILE_ENV: topo_file.name}
            ):
                querier = GroupTopologyQuerier()
        self.assertEqual(querier.query("10.0.0.1"), ("", "SG1"))
        self.assertEqual(querier.query("10.0.0.2"), ("", "SG1"))

    def test_inline_env_takes_precedence_over_file(self):
        file_topology = {"10.0.0.1": "SG1"}
        inline_topology = {"10.0.0.1": "SG2"}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as topo_file:
            json.dump(file_topology, topo_file)
            topo_file.flush()
            with patch.dict(
                os.environ,
                {
                    GROUP_TOPOLOGY_FILE_ENV: topo_file.name,
                    GROUP_TOPOLOGY_ENV: json.dumps(inline_topology),
                },
            ):
                querier = GroupTopologyQuerier()
        self.assertEqual(querier.query("10.0.0.1"), ("", "SG2"))

    def test_missing_topology_reports_empty_psw(self):
        querier = GroupTopologyQuerier()
        self.assertEqual(querier.query("10.0.0.1"), ("", ""))

    def test_invalid_json_fails_to_start(self):
        with patch.dict(os.environ, {GROUP_TOPOLOGY_ENV: "{not json"}):
            with self.assertRaisesRegex(ValueError, "Invalid cluster"):
                GroupTopologyQuerier()

    def test_non_dict_topology_fails_to_start(self):
        with patch.dict(os.environ, {GROUP_TOPOLOGY_ENV: '["SG1"]'}):
            with self.assertRaisesRegex(ValueError, "must be a JSON dict"):
                GroupTopologyQuerier()
        with patch.dict(os.environ, {GROUP_TOPOLOGY_ENV: '{"ip": 1}'}):
            with self.assertRaisesRegex(ValueError, "must be a JSON dict"):
                GroupTopologyQuerier()


class GroupSequenceTest(unittest.TestCase):
    def test_ep_dp_pp_contiguous(self):
        self.assertEqual(
            _ep_dp_pp_group_sequence({0: 2, 1: 3}), [0, 0, 1, 1, 1]
        )
        # arbitrary (non-contiguous) group ids are laid out ascending
        self.assertEqual(_ep_dp_pp_group_sequence({7: 1, 2: 2}), [2, 2, 7])

    def test_ep_pp_dp_aligned_keeps_slots_intra_group(self):
        # 2 groups x 4 nodes, pp=2, ep_workers=2: every 2-node EP slot
        # stays inside one group.
        self.assertEqual(
            _ep_pp_dp_group_sequence({0: 4, 1: 4}, pp=2, ep_workers=2),
            [0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_ep_pp_dp_interleaved_is_optimal_already(self):
        # The stage-major flatten interleaves one EP slot of every
        # column per stage, so ep_workers=1 with 2 groups alternates.
        self.assertEqual(
            _ep_pp_dp_group_sequence({0: 2, 1: 2}, pp=2, ep_workers=1),
            [0, 1, 0, 1],
        )

    def test_ep_pp_dp_unaligned_tolerates_crumbs_and_leftover(self):
        # {0:3, 1:3} with a 2-node slot: 1+1 whole slots fill the single
        # column, then the 1+1 crumbs complete the world pod by pod; the
        # last slot straddles, which the rerank reports.
        self.assertEqual(
            _ep_pp_dp_group_sequence({0: 3, 1: 3}, pp=2, ep_workers=2),
            [0, 0, 1, 1, 0, 1],
        )
        # N % pp remainder: {0:2, 1:3} with 1-node slots leaves 1 whole
        # slot the column inventory cannot place; it is appended slot by
        # slot after the columns instead of failing.
        self.assertEqual(
            _ep_dp_pp_group_sequence({0: 2, 1: 3}), [0, 0, 1, 1, 1]
        )
        self.assertEqual(
            _ep_pp_dp_group_sequence({0: 2, 1: 3}, pp=2, ep_workers=1),
            [1, 0, 1, 0, 1],
        )

    def test_ep_pp_dp_fills_ragged_columns_with_crumb_slots(self):
        # 4 groups x 3 nodes, pp=3 and ep_workers=2: only 4 whole slots
        # for the 2x3 matrix, so the greedy leaves column 1 ragged (1
        # of 3 slots). The 1+1+1+1 crumbs are packed into 2 straddling
        # EP slots that FILL column 1's rows 1-2 inside the matrix
        # (instead of being appended at the world tail), so every
        # stage row of the flatten keeps the full 2-slot width and
        # stays aligned with the uniform ep-dp-pp rank grid.
        self.assertEqual(
            _ep_pp_dp_group_sequence(
                {0: 3, 1: 3, 2: 3, 3: 3}, pp=3, ep_workers=2
            ),
            [0, 0, 3, 3, 1, 1, 0, 1, 2, 2, 2, 3],
        )

    def test_ep_pp_dp_fills_empty_columns_with_crumb_slots(self):
        # {0:5, 1:3, 2:3, 3:1} with pp=2, ep_workers=2: group 0 owns
        # column 0 (its whole 2 slots) and groups 1-2 share column 1,
        # leaving column 2 EMPTY (no whole slots left). The four crumbs
        # fill BOTH rows of the empty column with straddling EP slots,
        # so the matrix stays 3 columns wide at every stage row.
        self.assertEqual(
            _ep_pp_dp_group_sequence(
                {0: 5, 1: 3, 2: 3, 3: 1}, pp=2, ep_workers=2
            ),
            [0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 2, 3],
        )

    def test_ep_pp_dp_crumb_pairing_minimizes_slot_span(self):
        # 8 psw groups sized 3/3/3/2/2/1/1/1 nodes with ep_workers=4:
        # no whole slots at all, so both columns are filled entirely
        # with crumb slots. The pairing takes the largest crumb pile
        # and looks for an exact complementary pile first (3+1, 3+1,
        # 3+1), then pairs the two 2-node leftovers (2+2), so every
        # EP slot straddles exactly two psws; the flat group-id order
        # would instead tile [0,0,0,1], [1,1,2,2], [3,3,4,5] and
        # [6,10,10,10] with up to three psws per slot.
        sequence = _ep_pp_dp_group_sequence(
            {0: 3, 1: 3, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 10: 3},
            pp=2,
            ep_workers=4,
        )
        self.assertEqual(
            sequence,
            [0, 0, 0, 4, 10, 10, 10, 6, 1, 1, 1, 5, 2, 2, 3, 3],
        )
        for start in range(0, len(sequence), 4):
            self.assertEqual(len(set(sequence[start : start + 4])), 2)


class GroupTopologySorterTest(unittest.TestCase):
    def test_ep_dp_pp_groups_are_contiguous(self):
        # Creation order interleaves the psws; after the rerank every
        # psw occupies a contiguous world range (ascending psw order)
        # and the keys are the original node ranks.
        nodes = _nodes(
            {
                0: "SG1",
                1: "SG1",
                2: "SG2",
                3: "SG2",
                4: "SG1",
                5: "SG2",
            }
        )
        sorter = GroupTopologySorter(strategy=NodeGroupStrategy.EP_DP_PP)
        with self.assertLogs("dlrover.logger", level="INFO") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1, 4, 2, 3, 5])
        self.assertEqual(set(sorted_nodes.keys()), set(nodes.keys()))
        for node_rank, meta in sorted_nodes.items():
            self.assertIs(meta, nodes[node_rank])
        # every log line of the rerank is greppable with RERANK
        self.assertTrue(
            any("RERANK: node node_rank=" in log for log in logs.output)
        )
        self.assertTrue(
            any("RERANK: order: world position" in log for log in logs.output)
        )

    def test_ep_dp_pp_unknown_psw_is_a_peer_group(self):
        # Nodes whose IP is not in the cluster topology (empty psw) form
        # a DEFAULT group that is a peer of the known groups: under the
        # ascending-psw ordering "" sorts first, so the default group
        # occupies the first contiguous range like any other group
        # instead of being pushed to the tail.
        nodes = _nodes({0: "SG2", 1: "", 2: "SG1", 3: "SG2"})
        sorter = GroupTopologySorter(strategy=NodeGroupStrategy.EP_DP_PP)
        sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [1, 2, 0, 3])

    def test_ep_pp_dp_aligned_world(self):
        # 8 nodes: ranks 0-3 are SG1 and 4-7 are SG2, pp=2 and
        # ep_workers=2 (etp*ep=16, R=8). The layout keeps every EP slot
        # and every PP column intra-psw.
        nodes = _nodes(
            {
                0: "SG1",
                1: "SG1",
                2: "SG1",
                3: "SG1",
                4: "SG2",
                5: "SG2",
                6: "SG2",
                7: "SG2",
            }
        )
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            etp=2,
            ranks_per_node=8,
        )
        with self.assertLogs("dlrover.logger", level="INFO") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(
            list(sorted_nodes.keys()), [0, 1, 4, 5, 2, 3, 6, 7]
        )
        # no EP slot straddles the psw groups
        self.assertFalse(any("straddle" in log for log in logs.output))

    def test_ep_pp_dp_interleaved_creation_order(self):
        # The creation order already alternates the psws; with
        # ep_workers=1 and pp=2 the ep_pp_dp layout is optimal as-is,
        # which documents that the rerank is a no-op here.
        nodes = _nodes({0: "SG1", 1: "SG2", 2: "SG1", 3: "SG2"})
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            ranks_per_node=8,
        )
        sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1, 2, 3])

    def test_ep_pp_dp_reports_straddling_slots(self):
        # Unaligned group sizes ({0:3, 1:3} with a 2-node EP slot) are
        # tolerated; the layout reports the EP slots crossing groups.
        nodes = _nodes(
            {
                0: "SG1",
                1: "SG1",
                2: "SG1",
                3: "SG2",
                4: "SG2",
                5: "SG2",
            }
        )
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            etp=2,
            ranks_per_node=8,
        )
        with self.assertLogs("dlrover.logger", level="WARNING") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertEqual(len(sorted_nodes), 6)
        self.assertEqual(set(sorted_nodes.keys()), set(nodes.keys()))
        self.assertTrue(
            any("EP slot(s) straddle psw groups" in log for log in logs.output)
        )

    def test_ep_pp_dp_crumb_fill_keeps_pp_groups_intra_psw(self):
        # 12 nodes in 4 psws of sizes 5/3/3/1, pp=2 and
        # ep_workers=2 (etp*ep=16, R=8): group SG1 has enough whole
        # slots (2) to own the pure column 0. Group SG4 (1 node) and
        # the crumbs leave column 2 ragged; the crumbs FILL it with
        # straddling EP slots instead of being appended at the tail,
        # so the stage rows stay aligned with the uniform ep-dp-pp
        # rank grid and every PP group of the pure column 0 sits
        # within one psw.
        psws = ["SG1"] * 5 + ["SG2"] * 3 + ["SG3"] * 3 + ["SG4"]
        nodes = _nodes({rank: psw for rank, psw in enumerate(psws)})
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            etp=2,
            ranks_per_node=8,
        )
        with self.assertLogs("dlrover.logger", level="INFO") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(
            list(sorted_nodes.keys()),
            [0, 1, 5, 6, 2, 7, 3, 4, 8, 9, 10, 11],
        )
        # The PP communication groups of column 0 (ranks
        # stage*columns*ep_workers + 0 + node_in_slot in the
        # ep-dp-pp order) all sit within the single psw SG1.
        order = list(sorted_nodes.keys())
        columns, ep_workers, stages = 3, 2, 2
        for node_in_slot in range(ep_workers):
            line = [
                order[
                    stage * columns * ep_workers
                    + 0 * ep_workers
                    + node_in_slot
                ]
                for stage in range(stages)
            ]
            self.assertEqual(
                {psws[rank] for rank in line},
                {"SG1"},
            )
        # Only the 2 crumb-filled slots straddle: the layout reports
        # them instead of drifting the late stages into neighbor
        # columns.
        self.assertTrue(
            any("EP slot(s) straddle psw groups" in log for log in logs.output)
        )

    def test_single_group_keeps_the_rendezvous_order(self):
        # A single psw (or all-empty psws) leaves the world unchanged.
        nodes = _nodes({0: "SG1", 1: "SG1", 2: "SG1"})
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ranks_per_node=8,
        )
        sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1, 2])

        empty_nodes = _nodes({0: "", 1: ""})
        sorted_nodes = sorter.sort(empty_nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1])

    def test_unknown_strategy_keeps_the_rendezvous_order(self):
        nodes = _nodes({0: "SG1", 1: "SG2"})
        sorter = GroupTopologySorter(strategy="bogus")
        sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1])

    def test_ranks_per_node_falls_back_to_the_observed_process_num(self):
        # ranks_per_node unset: use the local world size observed at the
        # rendezvous (etp*ep=16, process_num=8 -> ep_workers=2).
        nodes = _nodes({0: "SG1", 1: "SG1", 2: "SG2", 3: "SG2"})
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            etp=2,
        )
        with self.assertLogs("dlrover.logger", level="INFO") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1, 2, 3])
        # the config log reports the observed ranks_per_node
        self.assertTrue(
            any(
                "ranks_per_node=8" in log and "RERANK: config" in log
                for log in logs.output
            )
        )

    def test_configured_ranks_per_node_mismatch_uses_the_observed_one(self):
        nodes = _nodes({0: "SG1", 1: "SG1", 2: "SG2", 3: "SG2"})
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ep=8,
            etp=2,
            ranks_per_node=4,
        )
        with self.assertLogs("dlrover.logger", level="WARNING") as logs:
            sorted_nodes = sorter.sort(nodes)
        self.assertTrue(
            any("differs from the observed" in log for log in logs.output)
        )
        # ep_workers=2 (from the observed R=8) is used for the layout
        self.assertEqual(len(sorted_nodes), 4)

    def test_missing_ranks_per_node_keeps_the_rendezvous_order(self):
        # Neither a configured ranks_per_node nor an observed
        # process_num: the rerank degrades to a no-op instead of
        # guessing the EP slot size.
        nodes = _nodes({0: "SG1", 1: "SG2"}, process_num=0)
        sorter = GroupTopologySorter(
            strategy=NodeGroupStrategy.EP_PP_DP,
            pp=2,
            ranks_per_node=None,
        )
        sorted_nodes = sorter.sort(nodes)
        self.assertListEqual(list(sorted_nodes.keys()), [0, 1])

    def test_sort_is_deterministic(self):
        # The same metas produce the same world order regardless of the
        # input dict iteration order, so re-freezing a later rendezvous
        # round does not reshuffle the ranks.
        psw_by_rank = {0: "SG2", 1: "SG1", 2: "SG2", 3: "SG1", 4: "SG2"}
        nodes_a = _nodes(psw_by_rank)
        nodes_b = _nodes({rank: psw_by_rank[rank] for rank in [4, 2, 0, 3, 1]})
        sorter = GroupTopologySorter(strategy=NodeGroupStrategy.EP_DP_PP)
        order_a = list(sorter.sort(nodes_a).keys())
        order_b = list(sorter.sort(nodes_b).keys())
        self.assertEqual(order_a, order_b)


if __name__ == "__main__":
    unittest.main()
