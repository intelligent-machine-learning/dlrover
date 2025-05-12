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

from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.tests.test_class import (
    TestActor,
    TestInteractiveTrainer,
)
from dlrover.python.rl.trainer.trainer import RoleGroupProxy


class BaseTrainerTest(unittest.TestCase):
    def test_basic(self):
        trainer = TestInteractiveTrainer(
            {RLRoleType.ACTOR: [None, None]},
            {RLRoleType.ACTOR: (TestActor, 1)},
            {"k1": "v1"},
        )
        self.assertIsNotNone(trainer)
        self.assertEqual(len(trainer.get_role_groups()), 1)
        self.assertTrue(isinstance(trainer.RG_ACTOR, RoleGroupProxy))
        self.assertEqual(len(trainer.RG_ACTOR._actor_handles), 2)

        self.assertEqual(len(trainer.actors), 2)
        self.assertEqual(len(trainer.rollouts), 0)
        self.assertEqual(len(trainer.get_role_groups()), 1)
        self.assertFalse(trainer.is_recoverable())
        self.assertEqual(len(trainer.actor_handles), 1)
        self.assertEqual(len(trainer.actor_handles[RLRoleType.ACTOR]), 2)
        self.assertEqual(len(trainer.actors), 2)
        self.assertEqual(len(trainer.rollouts), 0)
        self.assertEqual(len(trainer.rewards), 0)
        self.assertEqual(len(trainer.references), 0)
        self.assertEqual(len(trainer.critics), 0)
        self.assertEqual(trainer.actor_resource, 1)
        self.assertEqual(trainer.rollout_resource, 1)
        self.assertEqual(trainer.reference_resource, 1)
        self.assertEqual(trainer.reward_resource, 1)
        self.assertEqual(trainer.critic_resource, 1)
        self.assertEqual(len(trainer.config), 1)

    def test_role_group_proxy(self):
        role_group = RoleGroupProxy(RLRoleType.ACTOR, 2, TestActor, {}, [None])
        self.assertEqual(role_group.role, RLRoleType.ACTOR)
        self.assertEqual(role_group.world_size, 2)
        self.assertTrue(role_group._can_shard_invocation())
        with self.assertRaises(AttributeError):
            role_group.test0()
            role_group.test1()
            role_group.test2()
            role_group.test3()
            role_group.test4()
