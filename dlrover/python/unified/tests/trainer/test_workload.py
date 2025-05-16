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

from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.trainer.workload import BaseWorkload


class BaseWorkloadTest(unittest.TestCase):
    def tearDown(self):
        os.environ.clear()

    def test_basic(self):
        os.environ["NAME"] = "test"
        os.environ["ROLE"] = "ACTOR"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        workload = BaseWorkload(None, {"k1": "v1"})
        self.assertIsNotNone(workload)
        self.assertEqual(workload.name, "test")
        self.assertEqual(workload.role, RLRoleType.ACTOR)
        self.assertEqual(workload.rank, 0)
        self.assertEqual(workload.local_rank, 0)
        self.assertEqual(workload.world_size, 2)
        self.assertEqual(workload.local_world_size, 1)
        self.assertEqual(workload.torch_master_addr, "127.0.0.1")
        self.assertEqual(workload.torch_master_port, 29500)

        self.assertIsNone(workload._get_actor_id())

        self.assertTrue(workload.is_actor_role())
        self.assertFalse(workload.is_ref_role())
        self.assertFalse(workload.is_rollout_role())
        self.assertFalse(workload.is_reward_role())
        self.assertFalse(workload.is_critic_role())

        self.assertIsNone(workload.get_device_collocation())
        self.assertFalse(workload.has_device_collocation())
        self.assertFalse(workload.is_actor_or_rollout_device_collocation())
        self.assertIsNotNone(workload.get_runtime_info())

        workload.setup({"k2": "v2"})
        self.assertEqual(os.environ["k2"], "v2")
