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
from unittest import mock

from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.tests.base import BaseTest
from dlrover.python.unified.tests.test_class import (
    TestActor,
    TestInteractiveTrainer,
    TestRollout,
)
from dlrover.python.unified.trainer.trainer import (
    DefaultTrainer,
    RoleGroupProxy,
)


class BaseTrainerTest(BaseTest):
    def test_basic(self):
        trainer = TestInteractiveTrainer(
            {RLRoleType.ACTOR.name: [None, None]},
            {RLRoleType.ACTOR.name: (TestActor, 1)},
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
        self.assertEqual(len(trainer.actor_handles[RLRoleType.ACTOR.name]), 2)
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

    def test_default(self):
        trainer = DefaultTrainer(
            {RLRoleType.ACTOR.name: [None, None]},
            {RLRoleType.ACTOR.name: (TestActor, 1)},
            {"k1": "v1"},
        )
        trainer.init()
        trainer.fit()

    def test_role_group_proxy(self):
        role_group = RoleGroupProxy(
            RLRoleType.ACTOR.name, 2, TestActor, {}, [None]
        )
        self.assertEqual(role_group.role, RLRoleType.ACTOR.name)
        self.assertEqual(role_group.world_size, 2)
        self.assertTrue(role_group._can_shard_invocation())
        with self.assertRaises(AttributeError):
            role_group.test0()
        with self.assertRaises(AttributeError):
            role_group.test1()
        with self.assertRaises(AttributeError):
            role_group.test2()
        with self.assertRaises(AttributeError):
            role_group.test3()
        with self.assertRaises(AttributeError):
            role_group.test4()

        trainer = TestInteractiveTrainer(
            {RLRoleType.ACTOR.name: [], RLRoleType.ROLLOUT.name: []},
            {
                RLRoleType.ACTOR.name: (TestActor, 1),
                RLRoleType.ROLLOUT.name: (TestRollout, 1),
            },
            {},
        )
        self.assertIsNotNone(trainer)

        trainer.RG_ACTOR._actor_handles = mock.MagicMock(
            return_value=[mock.Mock()]
        )
        trainer.RG_ACTOR.test0()
        trainer.RG_ACTOR.test1()
        trainer.RG_ACTOR.test2()
        with self.assertRaises(Exception):
            trainer.RG_ACTOR.test3()
        with self.assertRaises(Exception):
            trainer.RG_ACTOR.test4()
