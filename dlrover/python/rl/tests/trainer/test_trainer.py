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


class TrainerTest(unittest.TestCase):
    def test_construct(self):
        trainer = TestInteractiveTrainer(
            {RLRoleType.ACTOR: [None, None]},
            {RLRoleType.ACTOR: TestActor},
            None,
        )
        self.assertIsNotNone(trainer)
        self.assertEqual(len(trainer.get_role_groups()), 1)
        self.assertTrue(isinstance(trainer.RG_ACTOR, RoleGroupProxy))
        self.assertEqual(len(trainer.RG_ACTOR._actor_handles), 2)

        self.assertEqual(len(trainer.actors), 2)
        self.assertEqual(len(trainer.rollouts), 0)
