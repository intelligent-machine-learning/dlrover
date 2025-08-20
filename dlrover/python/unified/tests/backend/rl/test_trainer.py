#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List
from unittest import mock
from unittest.mock import patch

from dlrover.python.unified.backend.rl.trainer import (
    RoleGroupProxy,
)
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.common.workload_base import ActorInfo, JobInfo
from dlrover.python.unified.common.workload_desc import CustomWorkloadDesc
from dlrover.python.unified.tests.backend.rl.classes import (
    TestActor,
    TestInteractiveTrainer,
)
from dlrover.python.unified.tests.base import BaseTest


class BaseTrainerTest(BaseTest):
    # Must be async, as ActorBase.__init__ expects an event loop to be running.
    @patch("ray.get_actor")
    @patch(
        "dlrover.python.unified.backend.rl.trainer.PrimeMasterApi.get_workers_by_role"
    )
    async def test_basic(self, mock_get_workers_by_role, mock_get_actor):
        spec = CustomWorkloadDesc(
            module_name=f"{__package__}.classes",
            class_name="TestInteractiveTrainer",
        )
        actor_spec = CustomWorkloadDesc(
            module_name=f"{__package__}.classes",
            class_name="TestActor",
        )
        mock_roles: Dict[str, List[ActorInfo]] = {
            "TRAINER": [ActorInfo(name="t", role="TRAINER", spec=spec)],
            "ACTOR": [
                ActorInfo(name="a0", role="ACTOR", spec=actor_spec),
                ActorInfo(name="a1", role="ACTOR", spec=actor_spec),
            ],
        }

        def get_workers_by_role(role: str, optional=False) -> List[ActorInfo]:
            if role not in mock_roles and not optional:
                raise ValueError(f"Role {role} not found")
            return mock_roles.get(role, [])

        mock_get_actor.return_value = "actor_handle"
        mock_get_workers_by_role.side_effect = get_workers_by_role

        trainer = TestInteractiveTrainer(
            JobInfo(name="test", job_id="test", user_config={"k1": "v1"}),
            ActorInfo(name="test", role="TRAINER", spec=spec),
        )
        self.assertIsNotNone(trainer)
        trainer._init_role_group_proxy()

        self.assertTrue(isinstance(trainer.RG_ACTOR, RoleGroupProxy))
        self.assertEqual(len(trainer.RG_ACTOR._actor_handles), 2)

        self.assertEqual(len(trainer.actors), 2)
        self.assertEqual(len(trainer.rollouts), 0)
        self.assertEqual(len(trainer.actors), 2)
        self.assertEqual(len(trainer.rollouts), 0)
        self.assertEqual(len(trainer.rewards), 0)
        self.assertEqual(len(trainer.references), 0)
        self.assertEqual(len(trainer.critics), 0)
        self.assertEqual(len(trainer.config), 1)

    @patch("ray.get_actor")
    @patch(
        "dlrover.python.unified.backend.rl.trainer.PrimeMasterApi.get_workers_by_role"
    )
    @patch("ray.wait")
    @patch("ray.get")
    async def test_role_group_proxy(
        self, patch_get, patch_wait, mock_get_workers_by_role, mock_get_actor
    ):
        patch_get.side_effect = lambda x: x
        patch_wait.side_effect = lambda *args, **kwargs: (args[0], [])

        role_group = RoleGroupProxy(
            RLRoleType.ACTOR.name, 2, TestActor, [None]
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

        spec = CustomWorkloadDesc(
            module_name="dlrover.python.unified.tests.test_class",
            class_name="TestInteractiveTrainer",
        )
        actor_spec = CustomWorkloadDesc(
            module_name="dlrover.python.unified.tests.test_class",
            class_name="TestActor",
        )
        mock_roles: Dict[str, List[ActorInfo]] = {
            "TRAINER": [ActorInfo(name="t", role="TRAINER", spec=spec)],
            "ACTOR": [
                ActorInfo(name="a0", role="ACTOR", spec=actor_spec),
                ActorInfo(name="a1", role="ACTOR", spec=actor_spec),
            ],
        }

        def get_workers_by_role(role: str, optional=False) -> List[ActorInfo]:
            if role not in mock_roles and not optional:
                raise ValueError(f"Role {role} not found")
            return mock_roles.get(role, [])

        mock_get_actor.return_value = "actor_handle"
        mock_get_workers_by_role.side_effect = get_workers_by_role

        trainer = TestInteractiveTrainer(
            JobInfo(name="test", job_id="test", user_config={}),
            ActorInfo(name="test", role="TRAINER", spec=spec),
        )
        self.assertIsNotNone(trainer)
        trainer._init_role_group_proxy()

        trainer.RG_ACTOR._actor_handles = mock.MagicMock(
            return_value=[mock.Mock()]
        )
        trainer.RG_ACTOR.test0()
        trainer.RG_ACTOR.test1()
        trainer.RG_ACTOR.test2()
        trainer.RG_ACTOR.test3()
        trainer.RG_ACTOR.test4()
