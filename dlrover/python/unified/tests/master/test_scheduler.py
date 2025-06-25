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
from unittest.mock import MagicMock

import ray

from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.unified.common.job_context import get_job_context
from dlrover.python.unified.master.graph import DLExecutionGraph
from dlrover.python.unified.master.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)
from dlrover.python.unified.tests.base import RayBaseTest
from dlrover.python.unified.tests.master.base import BaseMasterTest
from dlrover.python.unified.tests.test_data import TestData


class SimpleSchedulerTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        self.init_ray_safely(num_cpus=8)

    def test_schedule(self):
        graph = DLExecutionGraph(self._job_context.dl_context)
        scheduler = SimpleScheduler(graph)
        scheduler.get_master_actor_handle = MagicMock(return_value=None)

        for vertex in graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)

        scheduler.schedule()

        self.assertEqual(len(graph.get_all_actor_handles()), 4)
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


class GroupOrderedSchedulerSingleBundlePerNodeTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_NONE_COLOCATE_HOST_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = DLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = DLExecutionGraph(self._job_context.dl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)

        os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
        self.init_ray_safely(num_cpus=8)

    def tearDown(self):
        os.environ.clear()
        self.close_ray_safely()
        super().tearDown()

    def test_schedule(self):
        self.assertEqual(len(self.scheduler.graph.get_placement_group()), 0)
        self.scheduler.get_master_actor_handle = MagicMock(return_value=None)

        for vertex in self.graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)

        self.scheduler.schedule()

        self.assertEqual(
            len(self.graph.get_all_actor_handles()), 2 + 2 + 2 + 2
        )
        for vertex in self.graph.get_all_vertices():
            self.assertIsNotNone(vertex.actor_handle)

        self.assertIsNotNone(ray.get_actor("ACTOR_2-0_1-0"))
        self.assertIsNotNone(ray.get_actor("ROLLOUT_2-1_1-0"))

        self.scheduler.cleanup()
        with self.assertRaises(ValueError):
            ray.get_actor("ACTOR_2-0_1-0")

        for vertex in self.graph.get_all_vertices():
            self.assertIsNone(vertex.actor_handle)
            self.assertIsNone(vertex.pg)
            self.assertIsNone(vertex.pg_bundle_index)
            self.assertEqual(vertex.create_time, 0)

        self.assertFalse(bool(self.graph.get_placement_group()))


class GroupOrderedSchedulerSingleGroupPerNodeTest(RayBaseTest):
    def setUp(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_GROUPED_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = DLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, rl_context)

        self.graph = DLExecutionGraph(self._job_context.dl_context)
        self.scheduler = GroupOrderedScheduler(self.graph)

        self.init_ray_safely(num_cpus=8)

    def tearDown(self):
        os.environ.clear()
        self.close_ray_safely()

    def test_schedule(self):
        self.assertEqual(len(self.scheduler.graph.get_placement_group()), 0)
        self.scheduler.get_master_actor_handle = MagicMock(return_value=None)

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
