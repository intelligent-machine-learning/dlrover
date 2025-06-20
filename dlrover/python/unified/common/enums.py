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

from enum import Enum


class DLType(Enum):
    PRE = "PRE"
    SFT = "SFT"
    MULTIMODAL = "MULTIMODAL"
    RL = "RL"
    HYBRID = "HYBRID"


class DLStreamType(Enum):
    TASK_STREAM = "TASK_STREAM"
    DATA_STREAM = "DATA_STREAM"


class TrainerType(Enum):
    USER_DEFINED = "USER_DEFINED"
    GENERATED = "GENERATED"
    ELASTIC_TRAINING = "ELASTIC_TRAINING"


class InternalRoleType(Enum):
    ELASTIC = "ELASTIC"


class RLRoleType(Enum):
    ACTOR = "ACTOR"
    ROLLOUT = "ROLLOUT"
    REFERENCE = "REFERENCE"
    REWARD = "REWARD"
    CRITIC = "CRITIC"


class MasterStateBackendType(Enum):
    RAY_INTERNAL = "RAY_INTERNAL"
    HDFS = "HDFS"


class SchedulingStrategyType(Enum):
    AUTO = "AUTO"
    SIMPLE = "SIMPLE"
    GROUP = "GROUP"


class JobStage(Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    FAILOVER = "FAILOVER"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

    @classmethod
    def is_ending_stage(cls, stage) -> bool:
        if isinstance(stage, str):
            stage = JobStage[stage.upper()]
        return stage in (JobStage.FINISHED, JobStage.ERROR)


class FailoverLevel(Enum):
    GLOBAL = "GLOBAL"
    PARTIAL = "PARTIAL"
    IGNORE = "IGNORE"
