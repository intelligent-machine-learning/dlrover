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

from enum import Enum


class TrainerType(Enum):
    USER_DEFINED = ("USER_DEFINED", None, None)
    OPENRLHF_PPO_DEEPSPEED = ("OPENRLHF_PPO_DEEPSPEED", "PPO", "DEEPSPEED")

    def __init__(self, value, algorithm_type, arc_type):
        self._value = value
        self.algorithmType = algorithm_type
        self.arc_type = arc_type


class RLAlgorithmType(Enum):
    GRPO = "GRPO"
    PPO = "PPO"


class TrainerArcType(Enum):
    MEGATRON = "MEGATRON"
    FSDP = "FSDP"
    DEEPSPEED = "DEEPSPEED"


class RLRoleType(Enum):
    ACTOR = "ACTOR"
    GENERATOR = "GENERATOR"
    REFERENCE = "REFERENCE"
    REWARD = "REWARD"
    CRITIC = "CRITIC"


class MasterStateBackendType(Enum):
    RAY_INTERNAL = "RAY_INTERNAL"
    HDFS = "HDFS"
