# Copyright 2025 The EasyDL Authors. All rights reserved.
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

from dlrover.python.rl.common.enums import (
    MasterStateBackendType,
    ModelParallelismArcType,
    RLAlgorithmType,
    TrainerType,
)


class EnumsTest(unittest.TestCase):
    def test_trainer_type(self):
        self.assertTrue(TrainerType["USER_DEFINED"])
        self.assertIsNone(TrainerType["USER_DEFINED"].algorithmType)
        self.assertIsNone(TrainerType["USER_DEFINED"].arc_type)

        self.assertTrue(TrainerType["OPENRLHF_PPO_DEEPSPEED"])
        self.assertEqual(
            TrainerType["OPENRLHF_PPO_DEEPSPEED"].algorithmType,
            RLAlgorithmType.PPO.value,
        )
        self.assertEqual(
            TrainerType["OPENRLHF_PPO_DEEPSPEED"].arc_type,
            ModelParallelismArcType.DEEPSPEED.name,
        )

        with self.assertRaises(KeyError):
            self.assertTrue(RLAlgorithmType["TEST"])

    def test_rl_algorithm_type(self):
        self.assertTrue(RLAlgorithmType["GRPO"])
        self.assertTrue(RLAlgorithmType["PPO"])

        with self.assertRaises(KeyError):
            self.assertTrue(RLAlgorithmType["DPO"])

    def test_trainer_arc_type(self):
        self.assertTrue(ModelParallelismArcType["MEGATRON"])
        self.assertTrue(ModelParallelismArcType["FSDP"])
        self.assertTrue(ModelParallelismArcType["DEEPSPEED"])

        with self.assertRaises(KeyError):
            self.assertTrue(RLAlgorithmType["DCP"])

    def test_master_state_backend(self):
        self.assertTrue(MasterStateBackendType["RAY_INTERNAL"])
        self.assertTrue(MasterStateBackendType["HDFS"])

        with self.assertRaises(KeyError):
            self.assertTrue(MasterStateBackendType["LOCAL"])
