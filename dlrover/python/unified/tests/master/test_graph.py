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
import os
import time
from unittest.mock import MagicMock

from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.dl_context import DLContext, RLContext
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.master.graph import (
    DLExecutionEdge,
    DLExecutionGraph,
    FunctionInfo,
    VertexInvocationMeta,
)
from dlrover.python.unified.master.scheduler import GroupOrderedScheduler
from dlrover.python.unified.tests.base import RayBaseTest
from dlrover.python.unified.tests.test_class import TestActor, TestRollout
from dlrover.python.unified.tests.test_data import TestData


class ExecutionGraphTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        if self._testMethodName == "test_serialization":
            os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
            self.init_ray_safely(num_cpus=8)

    def tearDown(self):
        os.environ.clear()
        self.close_ray_safely()
        super().tearDown()

    def test_basic(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))

        graph = DLExecutionGraph(rl_context)
        self.assertIsNotNone(graph)
        self.assertIsNotNone(graph.dl_config)
        self.assertEqual(len(graph.get_all_vertices()), 1 + 1 + 1 + 1)
        self.assertEqual(len(graph.name_vertex_mapping), 1 + 1 + 1 + 1)
        self.assertEqual(len(graph.name_actor_mapping), 0)
        # not used for now
        self.assertEqual(len(graph.execution_edges), 0)

        actor_vertices = graph.get_vertices_by_role_type(RLRoleType.ACTOR.name)
        self.assertEqual(len(actor_vertices), 1)
        self.assertEqual(actor_vertices[0].role, RLRoleType.ACTOR.name)
        self.assertEqual(actor_vertices[0].name, "ACTOR_1-0_1-0")
        self.assertEqual(actor_vertices[0].get_cls(), TestActor)
        self.assertEqual(actor_vertices[0].rank, 0)
        self.assertEqual(actor_vertices[0].world_size, 1)

        rollout_vertex_0 = graph.get_vertex(RLRoleType.ROLLOUT.name, 0)
        self.assertEqual(rollout_vertex_0.role, RLRoleType.ROLLOUT.name)
        self.assertEqual(rollout_vertex_0.name, "ROLLOUT_1-0_1-0")
        self.assertEqual(rollout_vertex_0.get_cls(), TestRollout)
        self.assertEqual(rollout_vertex_0.rank, 0)
        self.assertEqual(rollout_vertex_0.world_size, 1)

        self.assertIsNotNone(
            graph.get_unit_resource_by_role(RLRoleType.ACTOR.name)
        )

        now = int(time.time())
        rollout_vertex_0.update_runtime_info(
            create_time=now, hostname="test.com", restart_count=2
        )
        self.assertEqual(rollout_vertex_0.create_time, now)
        self.assertEqual(rollout_vertex_0.exit_time, 0)
        self.assertEqual(rollout_vertex_0.hostname, "test.com")
        self.assertEqual(rollout_vertex_0.host_ip, "")
        self.assertEqual(rollout_vertex_0.restart_count, 2)

        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_1}",
        ]
        rl_context = DLContext.build_from_args(parse_job_args(args))

        graph = DLExecutionGraph(rl_context)
        self.assertIsNotNone(graph)
        self.assertIsNotNone(graph.dl_config)
        self.assertEqual(len(graph.get_all_vertices()), 2 + 2 + 1 + 1)
        self.assertEqual(len(graph.name_vertex_mapping), 2 + 2 + 1 + 1)
        self.assertEqual(len(graph.name_actor_mapping), 0)

        actor_vertices = graph.get_vertices_by_role_type(RLRoleType.ACTOR.name)
        self.assertEqual(len(actor_vertices), 2)
        self.assertEqual(actor_vertices[0].role, RLRoleType.ACTOR.name)
        self.assertEqual(actor_vertices[0].name, "ACTOR_2-0_1-0")
        self.assertEqual(actor_vertices[0].get_cls(), TestActor)
        self.assertEqual(actor_vertices[0].rank, 0)
        self.assertEqual(actor_vertices[0].world_size, 2)
        self.assertEqual(actor_vertices[0].local_rank, 0)
        self.assertEqual(actor_vertices[0].local_world_size, 1)
        self.assertEqual(actor_vertices[1].rank, 1)
        self.assertEqual(actor_vertices[1].world_size, 2)
        self.assertEqual(actor_vertices[1].local_rank, 0)
        self.assertEqual(actor_vertices[1].local_world_size, 1)

        rollout_vertices = graph.get_vertices_by_role_type(
            RLRoleType.ROLLOUT.name
        )
        self.assertEqual(rollout_vertices[0].rank, 0)
        self.assertEqual(rollout_vertices[0].world_size, 2)
        self.assertEqual(rollout_vertices[0].local_rank, 0)
        self.assertEqual(rollout_vertices[0].local_world_size, 1)
        self.assertEqual(rollout_vertices[1].rank, 1)
        self.assertEqual(rollout_vertices[1].world_size, 2)
        self.assertEqual(rollout_vertices[1].local_rank, 0)
        self.assertEqual(rollout_vertices[1].local_world_size, 1)

    def test_serialization(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_NONE_COLOCATE_HOST_GROUPED_RL_CONF}",
        ]
        rl_context = DLContext.build_from_args(parse_job_args(args))

        graph = DLExecutionGraph(rl_context)
        scheduler = GroupOrderedScheduler(graph)

        # add pg info
        scheduler.get_master_actor_handle = MagicMock(return_value=None)
        scheduler.schedule()
        for vertex in graph.get_all_vertices():
            self.assertIsNotNone(vertex.pg)
            self.assertIsNotNone(vertex.pg_bundle_index)

        # add runtime info
        create_time = int(time.time())
        hostname = "test.com"
        host_ip = "127.0.0.1"
        for vertex in graph.get_all_vertices():
            vertex.update_runtime_info(
                create_time=create_time,
                hostname=hostname,
                host_ip=host_ip,
            )

        # serialize and deserialize
        graph_bytes = graph.serialize()
        graph_recover = DLExecutionGraph.deserialize(graph_bytes)

        self.assertIsNotNone(graph_recover)
        self.assertEqual(
            len(graph.name_vertex_mapping),
            len(graph_recover.name_vertex_mapping),
        )
        self.assertEqual(graph.dl_config, graph_recover.dl_config)
        self.assertEqual(
            len(graph.get_placement_group()),
            len(graph_recover.get_placement_group()),
        )
        for vertex in graph.get_all_vertices():
            name = vertex.name
            self.assertEqual(vertex.role, graph.name_vertex_mapping[name].role)
            self.assertEqual(vertex.rank, graph.name_vertex_mapping[name].rank)
            self.assertEqual(
                vertex.world_size, graph.name_vertex_mapping[name].world_size
            )
            self.assertEqual(
                vertex.local_world_size,
                graph.name_vertex_mapping[name].local_world_size,
            )
            self.assertEqual(
                vertex.local_rank, graph.name_vertex_mapping[name].local_rank
            )
            self.assertEqual(
                vertex.module_name, graph.name_vertex_mapping[name].module_name
            )
            self.assertEqual(
                vertex.class_name, graph.name_vertex_mapping[name].class_name
            )
            self.assertEqual(
                vertex.resource, graph.name_vertex_mapping[name].resource
            )
            self.assertEqual(
                vertex.create_time, graph.name_vertex_mapping[name].create_time
            )
            self.assertEqual(
                vertex.hostname, graph.name_vertex_mapping[name].hostname
            )
            self.assertEqual(
                vertex.host_ip, graph.name_vertex_mapping[name].host_ip
            )
            self.assertEqual(vertex.pg, graph.name_vertex_mapping[name].pg)
            self.assertEqual(
                vertex.pg_bundle_index,
                graph.name_vertex_mapping[name].pg_bundle_index,
            )

    def test_vertex_invocation_meta(self):
        def test_input():
            pass

        function_info = FunctionInfo("test", test_input)
        self.assertIsNotNone(function_info)
        self.assertEqual(function_info.name, "test")

        vertex_invocation_meta = VertexInvocationMeta(
            {function_info.name: function_info}
        )
        self.assertIsNotNone(vertex_invocation_meta)
        self.assertEqual(
            vertex_invocation_meta.get_func("test"), function_info
        )

    def test_edge_basic(self):
        edge = DLExecutionEdge(0, RLRoleType.ACTOR, RLRoleType.ROLLOUT, "test")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.from_role, RLRoleType.ACTOR)
        self.assertEqual(edge.to_role, RLRoleType.ROLLOUT)
        self.assertEqual(edge.invocation_name, "test")
        self.assertIsNone(edge.async_group)
