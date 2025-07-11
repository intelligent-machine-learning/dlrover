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
from unittest.mock import patch

from dlrover.python.common.enums import ResourceType
from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.dl_context import (
    DLContext,
    RLContext,
    WorkloadGroupDesc,
)
from dlrover.python.unified.common.enums import RLRoleType, TrainerType
from dlrover.python.unified.common.exception import InvalidDLConfiguration
from dlrover.python.unified.tests.base import BaseTest
from dlrover.python.unified.tests.test_data import TestData


class DLContextTest(BaseTest):
    @patch(
        "dlrover.python.unified.common.dl_context"
        ".get_class_by_module_and_class_name"
    )
    def test_building(self, mock_get_class):
        # with valid input
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_MOCK_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        self.assertIsNotNone(rl_context)
        self.assertIsInstance(rl_context, RLContext)
        self.assertEqual(
            rl_context.trainer.trainer_type, TrainerType.USER_DEFINED
        )
        self.assertEqual(rl_context.config.get("c1"), "v1")
        self.assertEqual(rl_context.trainer.class_name, "TestTrainer")
        self.assertEqual(rl_context.trainer.module_name, "test_trainer")
        self.assertIsNone(rl_context.critic_workload)
        self.assertIsNotNone(rl_context.actor_workload)
        self.assertEqual(rl_context.actor_workload.instance_number, 1)
        self.assertEqual(rl_context.actor_workload.instance_resource.gpu, 0)
        self.assertEqual(rl_context.actor_workload.instance_resource.cpu, 1)
        self.assertEqual(rl_context.actor_workload.module_name, "test_actor")
        self.assertEqual(rl_context.actor_workload.class_name, "TestActor")
        self.assertTrue(rl_context.__str__())

        self.assertTrue(rl_context.validate())

        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_RL_CONF}",
        ]
        rl_context = DLContext.build_from_args(parse_job_args(args))
        self.assertTrue(rl_context.validate())

        # with invalid type(build failed)
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_DPO_MOCK_RL_CONF}",
        ]
        with self.assertRaises(InvalidDLConfiguration):
            DLContext.build_from_args(parse_job_args(args))

        # with invalid resource(validate failed)
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_INVALID_RESOURCE_RL_CONF_0}",
        ]
        rl_context = DLContext.build_from_args(parse_job_args(args))
        self.assertFalse(rl_context.validate())

    def test_serialization(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_MOCK_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        serialized = rl_context.serialize()
        self.assertIsNotNone(serialized)

        deserialized = RLContext.deserialize(serialized)
        self.assertIsNotNone(deserialized)
        self.assertEqual(deserialized.config.get("c1"), "v1")

    def test_workload_group_desc_build(self):
        # all spread
        roles = {
            RLRoleType.ACTOR.name: 8,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 4,
        }
        resource = {
            RLRoleType.ACTOR.name: 1,
            RLRoleType.ROLLOUT.name: 1,
            RLRoleType.CRITIC.name: 1,
        }
        desc = WorkloadGroupDesc.build(
            [], roles, resource, 4, unit=ResourceType.CPU
        )
        self.assertIsNotNone(desc)
        self.assertEqual(desc.capacity, 4)
        self.assertEqual(desc.unit, ResourceType.CPU)
        self.assertEqual(len(desc.groups), 3)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ACTOR.name), 4)
        self.assertEqual(desc.groups[0][1], 2)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.ROLLOUT.name), 4)
        self.assertEqual(desc.groups[1][1], 1)
        self.assertEqual(desc.groups[2][0].get(RLRoleType.CRITIC.name), 4)
        self.assertEqual(desc.groups[2][1], 1)
        self.assertFalse(desc.has_device_collocate())
        self.assertEqual(len(desc.split_groups_in_dict()[1]), 3)
        self.assertTrue(desc.validate())

        # some grouped
        groups = [{"actor": 2, "rollout": 2}]
        roles = {
            RLRoleType.ACTOR.name: 4,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 4,
        }
        desc = WorkloadGroupDesc.build(
            groups, roles, resource, 4, unit=ResourceType.CPU
        )
        self.assertIsNotNone(desc)
        self.assertEqual(desc.capacity, 4)
        self.assertEqual(desc.unit, ResourceType.CPU)
        self.assertEqual(len(desc.groups), 2)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ACTOR.name), 2)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ROLLOUT.name), 2)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.CRITIC.name), 4)
        self.assertTrue(desc.validate())

        roles = {
            RLRoleType.ACTOR.name: 8,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 4,
        }
        with self.assertRaises(AssertionError):
            WorkloadGroupDesc.build(
                groups, roles, resource, 4, ResourceType.GPU
            )

        groups = [{"actor": 2, "rollout": 2}, {"critic": 3, "reference": 1}]
        roles = {
            RLRoleType.ACTOR.name: 4,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 3,
            RLRoleType.REWARD.name: 4,
            RLRoleType.REFERENCE.name: 1,
        }
        resource = {
            RLRoleType.ACTOR.name: 1,
            RLRoleType.ROLLOUT.name: 1,
            RLRoleType.CRITIC.name: 1,
            RLRoleType.REFERENCE.name: 1,
            RLRoleType.REWARD.name: 1,
        }
        desc = WorkloadGroupDesc.build(
            groups, roles, resource, 4, ResourceType.GPU
        )
        self.assertEqual(len(desc.groups), 3)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ACTOR.name), 2)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ROLLOUT.name), 2)
        self.assertEqual(desc.groups[0][1], 2)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.CRITIC.name), 3)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.REFERENCE.name), 1)
        self.assertEqual(desc.groups[1][1], 1)
        self.assertEqual(desc.groups[2][0].get(RLRoleType.REWARD.name), 4)
        self.assertEqual(desc.groups[2][1], 1)
        self.assertTrue(desc.validate())

        # all grouped
        roles = {
            RLRoleType.ACTOR.name: 4,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 3,
            RLRoleType.REFERENCE.name: 1,
        }
        desc = WorkloadGroupDesc.build(
            groups, roles, resource, 4, ResourceType.GPU
        )
        self.assertEqual(len(desc.groups), 2)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ACTOR.name), 2)
        self.assertEqual(desc.groups[0][0].get(RLRoleType.ROLLOUT.name), 2)
        self.assertEqual(desc.groups[0][1], 2)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.CRITIC.name), 3)
        self.assertEqual(desc.groups[1][0].get(RLRoleType.REFERENCE.name), 1)
        self.assertEqual(desc.groups[1][1], 1)
        self.assertTrue(desc.validate())

        groups = [{"actor": 2, "rollout": 2}, {"actor": 1, "reference": 1}]
        roles = {
            RLRoleType.ACTOR.name: 4,
            RLRoleType.ROLLOUT.name: 4,
            RLRoleType.CRITIC.name: 3,
            RLRoleType.REFERENCE.name: 4,
        }
        with self.assertRaises(AssertionError):
            desc = WorkloadGroupDesc.build(
                groups, roles, resource, 4, ResourceType.GPU
            )
            self.assertFalse(desc.validate())

    def test_workload_group_resolve_and_validate(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_GROUPED_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))

        self.assertEqual(
            len(rl_context.workload_group.groups),
            3,
        )
        self.assertTrue(rl_context.validate())

        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_HOST_INVALID_GROUPED_RL_CONF_0}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        self.assertFalse(rl_context.validate())

        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_HOST_INVALID_GROUPED_RL_CONF_1}",
        ]
        rl_context = DLContext.build_from_args(parse_job_args(args))
        self.assertFalse(rl_context.validate())

    def test_validation(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_MOCK_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        rl_context._trainer = None
        self.assertFalse(rl_context.validate())

        rl_context = RLContext.build_from_args(parse_job_args(args))
        rl_context._node_number = 0
        self.assertFalse(rl_context.validate())

        rl_context = RLContext.build_from_args(parse_job_args(args))
        rl_context._device_per_node = 0
        self.assertFalse(rl_context.validate())

        rl_context = RLContext.build_from_args(parse_job_args(args))
        rl_context._trainer._torch_master_port = [1, 2, 3, 4, 5, 6, 7]
        self.assertFalse(rl_context.validate())

        rl_context = RLContext.build_from_args(parse_job_args(args))
        rl_context._trainer._module_name = "test"
        self.assertFalse(rl_context.validate())

        rl_context = RLContext.build_from_args(parse_job_args(args))

        rl_context.workload_group.cal_total_resource = mock.MagicMock(1)
        self.assertFalse(rl_context.validate())
