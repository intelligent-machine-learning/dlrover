# Copyright 2025 The DLRover Authors. All rights reserved.
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
import time
from typing import Dict, List, Tuple

from dlrover.python.common.resource import Resource
from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.graph import (
    PlacementGroupAllocation,
    RLExecutionGraph,
)
from dlrover.python.rl.tests.master.base import BaseMasterTest
from dlrover.python.rl.tests.test_class import TestActor, TestRollout
from dlrover.python.rl.tests.test_data import TestData


class ExecutionGraphTest(BaseMasterTest):
    def test_basic(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))

        graph = RLExecutionGraph(rl_context)
        self.assertIsNotNone(graph)
        self.assertIsNotNone(graph.rl_config)
        self.assertEqual(len(graph.get_all_vertices()), 2 + 2 + 1 + 1)
        self.assertEqual(len(graph.name_vertex_mapping), 2 + 2 + 1 + 1)
        self.assertEqual(len(graph.name_actor_mapping), 0)

        actor_vertices = graph.get_vertices_by_role_type(RLRoleType.ACTOR)
        self.assertEqual(len(actor_vertices), 2)
        self.assertEqual(actor_vertices[0].role, RLRoleType.ACTOR)
        self.assertEqual(
            actor_vertices[0].name, RLRoleType.ACTOR.value + "-" + str(0)
        )
        self.assertEqual(actor_vertices[0].get_cls(), TestActor)
        self.assertEqual(actor_vertices[0].rank, 0)
        self.assertEqual(actor_vertices[0].world_size, 2)
        self.assertFalse(actor_vertices[0].use_pg())
        self.assertEqual(actor_vertices[1].rank, 1)
        self.assertEqual(actor_vertices[1].world_size, 2)

        rollout_vertex_0 = graph.get_vertex(RLRoleType.ROLLOUT, 0)
        self.assertEqual(rollout_vertex_0.role, RLRoleType.ROLLOUT)
        self.assertEqual(
            rollout_vertex_0.name, RLRoleType.ROLLOUT.value + "-" + str(0)
        )
        self.assertEqual(rollout_vertex_0.get_cls(), TestRollout)
        self.assertEqual(rollout_vertex_0.rank, 0)
        self.assertEqual(rollout_vertex_0.world_size, 1)

        now = int(time.time())
        rollout_vertex_0.update_runtime_info(
            create_time=now, hostname="test.com", restart_count=2
        )
        self.assertEqual(rollout_vertex_0.create_time, now)
        self.assertEqual(rollout_vertex_0.exit_time, 0)
        self.assertEqual(rollout_vertex_0.hostname, "test.com")
        self.assertEqual(rollout_vertex_0.host_ip, "")
        self.assertEqual(rollout_vertex_0.restart_count, 2)

    def test_graph_with_pg(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_HOST_GROUPED_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))

        graph = RLExecutionGraph(rl_context)
        self.assertIsNotNone(graph)

        vertices = graph.get_vertices_by_role_type(RLRoleType.ACTOR)
        self.assertTrue(vertices[0].use_pg())
        self.assertFalse(vertices[0].is_pg_allocated())

        vertices = graph.get_vertices_by_role_type(RLRoleType.ROLLOUT)
        self.assertTrue(vertices[0].use_pg())
        self.assertFalse(vertices[0].is_pg_allocated())

        vertices = graph.get_vertices_by_role_type(RLRoleType.REWARD)
        self.assertFalse(vertices[0].use_pg())
        self.assertFalse(vertices[0].is_pg_allocated())

        vertices = graph.get_vertices_by_role_type(RLRoleType.REFERENCE)
        self.assertFalse(vertices[0].use_pg())
        self.assertFalse(vertices[0].is_pg_allocated())

        self.assertNotEqual(
            graph.get_all_vertices_with_pg_priority(), graph.get_all_vertices()
        )

    def test_pg_allocation(self):
        resource0 = Resource(gpu=1)
        resource1 = Resource(gpu=2)
        bundles: Dict[RLRoleType, List[Tuple[int, Resource]]] = {
            RLRoleType.ACTOR: [(0, resource0), (1, resource0)],
            RLRoleType.ROLLOUT: [(2, resource1)],
        }
        pg_allocation = PlacementGroupAllocation("test", 0, "PACK", bundles)

        self.assertIsNotNone(pg_allocation)
        self.assertIsNone(pg_allocation._instance)
        self.assertEqual(
            pg_allocation._get_bundle_resource(),
            [{"GPU": 1}, {"GPU": 1}, {"GPU": 2}],
        )
        self.assertEqual(pg_allocation.get_bundle_size(), 3)
        self.assertEqual(
            pg_allocation.get_bundle_index_by_vertex_name("a1"), -1
        )
        self.assertFalse(pg_allocation.is_full())

        self.assertEqual(pg_allocation.allocate(RLRoleType.ROLLOUT, "r0"), 2)
        self.assertEqual(pg_allocation.allocate(RLRoleType.ROLLOUT, "r1"), -1)
        self.assertFalse(pg_allocation.is_full())
        self.assertEqual(pg_allocation.allocate(RLRoleType.ACTOR, "a0"), 0)
        self.assertEqual(pg_allocation.allocate(RLRoleType.ACTOR, "a1"), 1)
        self.assertEqual(pg_allocation.allocate(RLRoleType.ACTOR, "a3"), -1)
        self.assertTrue(pg_allocation.is_full())
