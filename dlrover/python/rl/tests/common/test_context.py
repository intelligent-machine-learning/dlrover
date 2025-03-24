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
from unittest.mock import patch

from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.context import RLContext
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    TrainerArcType,
    TrainerType,
)
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.rl.tests.test_data import TestData


class RLContextTest(unittest.TestCase):
    @patch(
        "dlrover.python.rl.common.context.get_class_by_module_and_class_name"
    )
    def test_building(self, mock_get_class):
        # with valid input
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_MOCK_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        self.assertIsNotNone(rl_context)
        self.assertEqual(
            rl_context.trainer.trainer_type, TrainerType.USER_DEFINED
        )
        self.assertEqual(
            rl_context.trainer.trainer_arc_type, TrainerArcType.MEGATRON
        )
        self.assertEqual(
            rl_context.trainer.algorithm_type, RLAlgorithmType.GRPO
        )
        self.assertEqual(rl_context.trainer.config.get("c1"), "v1")
        self.assertEqual(rl_context.trainer.class_name, "TestTrainer")
        self.assertEqual(rl_context.trainer.module_name, "test_trainer")
        self.assertIsNone(rl_context.critic_workload)
        self.assertIsNotNone(rl_context.actor_workload)
        self.assertEqual(rl_context.actor_workload.instance_number, 2)
        self.assertEqual(rl_context.actor_workload.instance_resource, {})
        self.assertEqual(rl_context.actor_workload.module_name, "test_actor")
        self.assertEqual(rl_context.actor_workload.class_name, "TestActor")
        self.assertTrue(rl_context.__str__())

        self.assertTrue(rl_context.validate())
        self.assertTrue(rl_context.validate())

        # with invalid input(build failed)
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_DPO_MOCK_RL_CONF}",
        ]
        with self.assertRaises(InvalidRLConfiguration):
            RLContext.build_from_args(parse_job_args(args))

    def test_serialization(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_MOCK_RL_CONF}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        serialized = rl_context.serialize()
        self.assertIsNotNone(serialized)

        deserialized = RLContext.deserialize(serialized)
        self.assertIsNotNone(deserialized)
        self.assertEqual(
            deserialized.trainer.algorithm_type, RLAlgorithmType.GRPO
        )
        self.assertEqual(deserialized.trainer.config.get("c1"), "v1")
