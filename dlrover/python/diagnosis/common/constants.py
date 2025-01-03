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
    LOGS = "logs"
    RANK = "rank"
    EXPIRED_TIME_PERIOD = "expired_tie_period"
    EXECUTABLE_TIME_PERIOD = "executable_time_period"

    EVENT_TYPE = "event_type"
    EVENT_INSTANCE = "event_instance"
    EVENT_ACTION = "event_action"
    EVENT_MSG = "event_msg"
    EVENT_LABELS = "event_labels"


class DiagnosisConstant(object):
    MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS = 180
    # the minimum diagnosis interval is 5 seconds
    AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS = 5
    AGENT_PERIODICALLY_REPORT_INTERVAL_SECS = 15
    MASTER_INSTANCE = -1
    ANY_INSTANCE = -2
    LOCAL_INSTANCE = -3
    ACTION_EXPIRED_TIME_PERIOD_DEFAULT = 60 * 5
    MAX_ACTION_QUEUE_SIZE = 1000


class DiagnosisErrorConstant(object):
    GPU_LOST = "GPU is lost"


class DiagnosisDataType(object):
    GENERIC = "GENERIC"
    TRAINING_LOG = "TRAINING_LOG"
    XPU_TIMER_METRIC = "XPU_TIMER_METRIC"


class DiagnosisActionType(object):
    # common
    NONE = "no_action"
    ANY = "any_action"
    LOG = "log"

    # master operation
    MASTER_RELAUNCH_WORKER = "master_relaunch_worker"
    EVENT = "event"

    # node operation
    RESTART_WORKER = "restart_worker"
    RELAUNCH_WORKER = "relaunch_worker"
