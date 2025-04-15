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
import unittest

from dlrover.python.common.enums import ResourceType
from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.graph import RLExecutionGraph
from dlrover.python.rl.master.placement import SingleBundlePerNodePlacement
from dlrover.python.rl.master.scheduler import GroupOrderedScheduler
from dlrover.python.rl.tests.test_data import TestData


class SingleBundlePerNodePlacementTest(unittest.TestCase):
    def setUp(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_NONE_COLOCATE_HOST_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = RLExecutionGraph(self._job_context.rl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)
        self.placement = self.scheduler.placement

        os.environ[RLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"

    def tearDown(self):
        os.environ.clear()

    def test_prepare_allocations(self):
        self.assertEqual(
            self.graph.rl_context.trainer.device_type.name,
            ResourceType.CPU.name,
        )

        self.placement.prepare_placement_group()

        self.assertEqual(len(self.scheduler.graph.get_placement_group()), 1)

        pg_allocation = self.scheduler.graph.get_placement_group(
            SingleBundlePerNodePlacement.PG_NAME
        )
        self.assertEqual(len(pg_allocation._bundles), 4)
        self.assertEqual(
            pg_allocation._bundles[0].get("CPU"),
            self._job_context.rl_context.trainer.device_per_node,
        )
        self.assertEqual(pg_allocation._strategy, "SPREAD")

        self.placement.allocate_placement_group()
        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ACTOR)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ROLLOUT)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REFERENCE)
        self.assertEqual(vertices[0].pg_bundle_index, 2)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REWARD)
        self.assertEqual(vertices[0].pg_bundle_index, 3)


class SingleGroupPerNodePlacementTest(unittest.TestCase):
    def setUp(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = RLExecutionGraph(self._job_context.rl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)
        self.placement = self.scheduler.placement

    def test_prepare_allocations(self):
        self.assertEqual(
            self.graph.rl_context.trainer.device_type.name,
            ResourceType.CPU.name,
        )

        self.placement.prepare_placement_group()

        self.assertEqual(
            len(self.scheduler.graph.get_placement_group()),
            self.graph.rl_context.trainer.node_number,
        )

        for pg_allocation in list(
            self.scheduler.graph.get_placement_group().values()
        ):
            self.assertEqual(
                len(pg_allocation._bundles),
                self.graph.rl_context.trainer.device_per_node,
            )
            self.assertEqual(
                pg_allocation._bundles[0].get("CPU"),
                1,
            )
            self.assertEqual(pg_allocation._strategy, "STRICT_PACK")

        self.placement.allocate_placement_group()

        for pg_allocation in list(
            self.scheduler.graph.get_placement_group().values()
        ):
            self.assertTrue(
                len(pg_allocation._allocation)
                >= self.graph.rl_context.trainer.device_per_node
            )
            self.assertTrue(pg_allocation.is_full())
            if pg_allocation.name == "SINGLE_GROUP_PER_NODE_0":
                self.assertTrue(
                    "ACTOR_4-0_2-0" in list(pg_allocation._allocation.keys())
                )
                self.assertTrue(
                    "ACTOR_4-1_2-1" in list(pg_allocation._allocation.keys())
                )
                self.assertTrue(
                    "ROLLOUT_4-0_2-0" in list(pg_allocation._allocation.keys())
                )
                self.assertTrue(
                    "ROLLOUT_4-1_2-1" in list(pg_allocation._allocation.keys())
                )
            elif pg_allocation.name == "SINGLE_GROUP_PER_NODE_2":
                self.assertTrue(
                    "REFERENCE_2-0_2-0"
                    in list(pg_allocation._allocation.keys())
                )
                self.assertTrue(
                    "REFERENCE_2-1_2-1"
                    in list(pg_allocation._allocation.keys())
                )

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ACTOR)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)
        self.assertEqual(vertices[2].pg_bundle_index, 0)
        self.assertEqual(vertices[3].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ROLLOUT)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)
        self.assertEqual(vertices[2].pg_bundle_index, 0)
        self.assertEqual(vertices[3].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REFERENCE)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REWARD)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 1)
