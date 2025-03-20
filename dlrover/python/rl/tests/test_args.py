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
import argparse
import unittest

from dlrover.python.rl.common.args import (
    _parse_algorithm_type,
    _parse_omega_config,
    _parse_trainer_arc_type,
    _parse_trainer_type,
)
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    TrainerArcType,
    TrainerType,
)


class ArgsTest(unittest.TestCase):
    def test_parse_trainer_type(self):
        self.assertEqual(
            _parse_trainer_type("user_defined"), TrainerType.USER_DEFINED
        )
        self.assertEqual(
            _parse_trainer_type("OPENRLHF_DEEPSPEED"),
            TrainerType.OPENRLHF_DEEPSPEED,
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_trainer_arc_type("test")

    def test_parse_algorithm_type(self):
        self.assertEqual(_parse_algorithm_type("GRPO"), RLAlgorithmType.GRPO)
        self.assertEqual(_parse_algorithm_type("grpo"), RLAlgorithmType.GRPO)
        self.assertEqual(_parse_algorithm_type("PPO"), RLAlgorithmType.PPO)
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_algorithm_type("DPO")

    def test_parse_trainer_arc_type(self):
        self.assertEqual(
            _parse_trainer_arc_type("megatron"), TrainerArcType.MEGATRON
        )
        self.assertEqual(_parse_trainer_arc_type("FSDP"), TrainerArcType.FSDP)
        self.assertEqual(
            _parse_trainer_arc_type("DEEPSPEED"), TrainerArcType.DEEPSPEED
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_trainer_arc_type("DCP")

    def test_parse_trainer_config(self):
        self.assertIsNotNone(_parse_omega_config({}))
        self.assertIsNotNone(_parse_omega_config("{}"))
        self.assertEqual(_parse_omega_config('{"k1": "v1"}').get("k1"), "v1")
        self.assertEqual(_parse_omega_config({"k1": "v1"}).get("k1"), "v1")
        self.assertEqual(
            _parse_omega_config({"k1": "v1", "k2": {"k22": "v2"}})
            .get("k2")
            .get("k22"),
            "v2",
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_omega_config(123)
