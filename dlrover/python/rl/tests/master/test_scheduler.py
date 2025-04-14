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
import unittest
from unittest.mock import MagicMock

import ray

from dlrover.python.common.enums import ResourceType
from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.graph import RLExecutionGraph
from dlrover.python.rl.master.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)
from dlrover.python.rl.tests.master.base import BaseMasterTest
from dlrover.python.rl.tests.test_data import TestData


class SimpleSchedulerTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        ray.init()

    def tearDown(self):
        super().tearDown()
        ray.shutdown()

    def test_schedule(self):
        graph = RLExecutionGraph(self._job_context.rl_context)
        scheduler = SimpleScheduler(graph)
        scheduler.get_master_actor_handle = MagicMock(return_value=None)

        for vertex in graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)

        scheduler.schedule()

        self.assertEqual(len(graph.get_all_actor_handles()), 6)
        for vertex in graph.get_all_vertices():
            self.assertIsNotNone(vertex.actor_handle)

        self.assertIsNotNone(ray.get_actor("ACTOR_1-0_1-0"))
        scheduler.cleanup()
        with self.assertRaises(ValueError):
            ray.get_actor("ACTOR_1-0_1-0")

        for vertex in graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)
            self.assertIsNone(vertex.pg)
            self.assertIsNone(vertex.pg_bundle_index)
            self.assertEqual(vertex.create_time, 0)


class GroupOrderedSchedulerMockTest(unittest.TestCase):
    def setUp(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_HOST_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = RLExecutionGraph(self._job_context.rl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)

    def test_prepare_allocations(self):
        self.assertEqual(
            self.graph.rl_context.trainer.device_type.name,
            ResourceType.CPU.name,
        )

        self.scheduler._prepare_placement_group()

        self.assertEqual(len(self.scheduler.graph.get_placement_group()), 1)

        pg_allocations = self.scheduler.graph.get_placement_group()[
            GroupOrderedScheduler.SINGLE_PG_NAME
        ]
        self.assertIsNotNone(pg_allocations)
        self.assertEqual(len(pg_allocations), 1)

        pg_allocation = pg_allocations[0]
        self.assertEqual(len(pg_allocation._bundles), 4)
        self.assertEqual(
            pg_allocation._bundles[0].get("CPU"),
            self._job_context.rl_context.trainer.device_per_node,
        )
        self.assertEqual(pg_allocation._strategy, "STRICT_SPREAD")

        self.scheduler._allocate_placement_group()
        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ACTOR)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 0)
        self.assertEqual(vertices[2].pg_bundle_index, 1)
        self.assertEqual(vertices[3].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.ROLLOUT)
        self.assertEqual(vertices[0].pg_bundle_index, 0)
        self.assertEqual(vertices[1].pg_bundle_index, 0)
        self.assertEqual(vertices[2].pg_bundle_index, 1)
        self.assertEqual(vertices[3].pg_bundle_index, 1)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REFERENCE)
        self.assertEqual(vertices[0].pg_bundle_index, 2)

        vertices = self.graph.get_vertices_by_role_type(RLRoleType.REWARD)
        self.assertEqual(vertices[0].pg_bundle_index, 3)


class GroupOrderedSchedulerTest(unittest.TestCase):
    def setUp(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_HOST_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = RLExecutionGraph(self._job_context.rl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)

        ray.init(num_cpus=9)

    def tearDown(self):
        ray.shutdown()

    def test_schedule(self):
        self.assertEqual(len(self.scheduler.graph.get_placement_group()), 0)
        self.scheduler.get_master_actor_handle = MagicMock(return_value=None)
        self.scheduler._get_placement_strategy = MagicMock(
            return_value="SPREAD"
        )

        for vertex in self.graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)

        self.scheduler.schedule()

        self.assertEqual(
            len(self.graph.get_all_actor_handles()), 4 + 4 + 2 + 2
        )
        for vertex in self.graph.get_all_vertices():
            self.assertIsNotNone(vertex.actor_handle)

        self.assertIsNotNone(ray.get_actor("ACTOR_4-0_2-0"))
        self.assertIsNotNone(ray.get_actor("ROLLOUT_4-3_2-1"))

        self.scheduler.cleanup()
        with self.assertRaises(ValueError):
            ray.get_actor("ACTOR_4-0_2-0")

        for vertex in self.graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)
            self.assertIsNone(vertex.pg)
            self.assertIsNone(vertex.pg_bundle_index)
            self.assertEqual(vertex.create_time, 0)

        self.assertFalse(bool(self.graph.get_placement_group()))
