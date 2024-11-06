# Copyright 2024 The DLRover Authors. All rights reserved.
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


class EnvConfigKey(object):
    XPU_TIMER_PORT = "XPU_TIMER_PORT"


class InferenceConfigKey(object):
    LOG_FILE = "log_file"
    ERRORS = "errors"


class DiagnosisConstant(object):
    MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS = 180
    AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS = 60
    MASTER = -1
    ANY_INSTANCE = -2
    LOCAL_INSTANCE = -3
    ACTION_EXPIRED_TIME_PERIOD_DEFAULT = 60 * 5


class DiagnosisDataType(object):
    GENERIC = "GENERIC"
    TRAINING_LOG = "TRAINING_LOG"
    XPU_TIMER_METRIC = "XPU_TIMER_METRIC"


class DiagnosisActionType(object):
    # common
    NONE = "no_action"
    ANY = "any_action"

    # master operation
    MASTER_RELAUNCH_WORKER = "master_relaunch_worker"
    EVENT = "event"

    # node operation
    NODE_ACT = "node_action"
    RESTART_WORKER = "restart_worker"
    RELAUNCH_WORKER = "relaunch_worker"
