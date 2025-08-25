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

import ray

from dlrover.python.master.watcher.ray_watcher import (
    ActorWatcher,
    parse_from_actor_name,
)
from dlrover.python.unified.common.constant import DLWorkloadEnv


class RayWatcherTest(unittest.TestCase):
    def test_parse_from_actor_name(self):
        self.assertEqual(parse_from_actor_name("vertex1"), "vertex1")
        self.assertEqual(parse_from_actor_name("vertex_1"), ("vertex", "1"))
        self.assertEqual(
            parse_from_actor_name("vertex_2-1_2-1"),
            ("vertex", "2", "1", "2", "1"),
        )


@ray.remote
class TestWorkload(object):
    def ping(self):
        return


class ActorWatcherTest(unittest.TestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()

    def test_list(self):
        watcher = ActorWatcher("test", "default")
        nodes = watcher.list(actor_class="TestWorkload")
        self.assertEqual(len(nodes), 0)

        actor_name = "ELASTIC_1-0_1-0"
        runtime_env = {
            "env_vars": {
                DLWorkloadEnv.JOB: "test",
                DLWorkloadEnv.NAME: actor_name,
                DLWorkloadEnv.ROLE: "ELASTIC",
                DLWorkloadEnv.RANK: "0",
                DLWorkloadEnv.WORLD_SIZE: "1",
                DLWorkloadEnv.LOCAL_RANK: "0",
                DLWorkloadEnv.LOCAL_WORLD_SIZE: "1",
            }
        }
        workload_handle = TestWorkload.options(
            name=actor_name, runtime_env=runtime_env
        ).remote()
        ray.get(workload_handle.ping.remote())

        nodes = watcher.list(actor_class="TestWorkload")
        self.assertEqual(len(nodes), 1)
