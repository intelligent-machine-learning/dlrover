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


class MasterStage(str, Enum):
    INIT = "INIT"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class DiagnosticInfoType(str, Enum):
    """Diagnostic info type of worker execution."""

    NORMAL = "NORMAL"
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"


class DiagnosticResponsibility(int, Enum):
    """Diagnostic responsibility of worker abnormal execution."""

    ROOT_CAUSE = 0  # confirm that it was caused by oneself.
    RELATED = 1  # confirm that it is related to oneself
    BE_AFFECTED = 2  # confirm that it has nothing to do with oneself
    UNKNOWN = 3


class ExecutionResultType(str, Enum):
    """Result type of worker execution."""

    SUCCESS = "SUCCESS"  # Finished successfully
    FAIL = "FAIL"  # Finished with failure


class ACCELERATOR_TYPE(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"


class DLStreamType(Enum):
    TASK_STREAM = "TASK_STREAM"
    DATA_STREAM = "DATA_STREAM"


class RLRoleType(str, Enum):
    TRAINER = "TRAINER"
    ACTOR = "ACTOR"
    ROLLOUT = "ROLLOUT"
    REFERENCE = "REFERENCE"
    REWARD = "REWARD"
    CRITIC = "CRITIC"


class MasterStateBackendType(Enum):
    RAY_INTERNAL = "RAY_INTERNAL"
    HDFS = "HDFS"
    IN_MEMORY = "IN_MEMORY"  # TEST Only, not for prod


class WorkloadEntrypointType(str, Enum):
    MODULE_FUNC = "MODULE_FUNC"
    PY_CMD = "PY_CMD"
