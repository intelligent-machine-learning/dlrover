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

from dlrover.python.rl.common.enums import RLAlgorithmType, TrainerArcType


class EnumsTest(unittest.TestCase):
    def test_rl_algorithm_type(self):
        self.assertTrue(RLAlgorithmType["GRPO"])
        self.assertTrue(RLAlgorithmType["PPO"])

        try:
            self.assertTrue(RLAlgorithmType["DPO"])
            self.fail()
        except KeyError:
            pass

    def test_trainer_arc_type(self):
        self.assertTrue(TrainerArcType["MEGATRON"])
        self.assertTrue(TrainerArcType["FSDP"])
        self.assertTrue(TrainerArcType["DEEPSPEED"])

        try:
            self.assertTrue(RLAlgorithmType["DCP"])
            self.fail()
        except KeyError:
            pass
