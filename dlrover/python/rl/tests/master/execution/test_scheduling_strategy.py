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
from unittest.mock import MagicMock

import ray

from dlrover.python.rl.master.execution.graph import RLExecutionGraph
from dlrover.python.rl.master.execution.scheduling_strategy import (
    SimpleStrategy,
)
from dlrover.python.rl.tests.master.base import BaseMasterTest


class SimpleStrategyTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        ray.init()

    def tearDown(self):
        super().tearDown()
        ray.shutdown()

    def test_schedule(self):
        graph = RLExecutionGraph(self._job_context.rl_context)
        strategy = SimpleStrategy()
        strategy.get_master_actor_handle = MagicMock(return_value=None)

        strategy.schedule(graph)

        self.assertEqual(len(graph.get_all_actor_handles()), 6)
        for vertex in graph.get_all_vertices():
            self.assertTrue(vertex.create_time)
            self.assertTrue(vertex.hostname)
            self.assertTrue(vertex.host_ip)

        self.assertIsNotNone(ray.get_actor("ACTOR-0"))
        strategy.cleanup(graph)
        with self.assertRaises(ValueError):
            ray.get_actor("ACTOR-0")
