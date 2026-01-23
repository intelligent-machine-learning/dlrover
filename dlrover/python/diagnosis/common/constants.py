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
    METRIC_COLLECT_INTERVAL_SECS = 60
    CHECK_TENSOR_DEFAULT_RECORDS = 30

    AGENT_PERIODICALLY_REPORT_INTERVAL_SECS = 15
    MASTER_INSTANCE = -1
    ANY_INSTANCE = -2
    LOCAL_INSTANCE = -3
    ACTION_EXPIRED_TIME_PERIOD_DEFAULT = 60 * 5
    MAX_ACTION_QUEUE_SIZE = 1000

    MIN_DIAGNOSIS_INTERVAL = 15


class DiagnosticianType(object):
    NODE_FAILURE = "node_failure"
    RESOURCE_COLLECT_FAILURE = "resource_collection_failure"
    NODE_INCONSISTENCY = "node_inconsistency"
    TRAINING_HANG = "training_hang"


class DiagnosisErrorConstant(object):
    GPU_LOST = "GPU is lost"
    PRE_CHECK_FAILED = "Pre-check failed"
    NODE_FAILED = "Node failed"
    REPEATED_NODE = "Repeated node"
    TRAINING_IS_HANG = "Training is hang"


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
    JOB_ABORT = "job_abortion"
    JOB_RESTART = "job_restart"
    MASTER_RELAUNCH_WORKER = "master_relaunch_worker"
    EVENT = "event"

    # node operation
    RESTART_WORKER = "restart_worker"
    RELAUNCH_WORKER = "relaunch_worker"

    # job operation
    RESTART_JOB = "restart_job"
    ABORT_JOB = "abort_job"


class DiagnosisResult(object):
    # diag invalid param
    DIAG_INVALID_PARAM = "invalid"
    # diag error
    DIAG_ERROR = "error"
    # waiting for more data to finish diag
    DIAG_WAITING = "waiting"
    # continue to next diagnosis phase
    DIAG_CONTINUE = "continue"
    # diag finished, job is healthy
    DIAG_HEALTHY = "succeeded"
    # diag finished, job is hang
    DIAG_HANG = "hang"
    # diag finished, job has straggler
    DIAG_STRAGGLE = "straggle"
    # diag finished, job has failure node
    DIAG_FAILURE = "failure"
    # diag finished, job has failure which cause abortion
    DIAG_ABORT = "abort"


class JobHangPolicy(object):
    XPU_TIMER = "xpu_timer"
    STEP_HANG = "step_hang"
    CKPT_HANG = "ckpt_hang"
    TENSOR_ZERO = "tensor_zero"
    NPU_ZERO = "npu_zero"
    NVLINK_DROP = "nvlink_drop"
    RDMA_DROP = "rdma_drop"
    PROCESS_HANG = "process_hang"


class JobHangWatermark(object):
    # TENSOR_UTIL is [0, 1]
    TENSOR_UTIL_LOW_WM = 0.001
    TENSOR_UTIL_HIGH_WM = 0.8
    # GPU_UTIL is [0, 100]
    GPU_UTIL_LOW_WM = 0.5
    GPU_UTIL_HIGH_WM = 98
    # NPU_UTIL is [0, 100]
    NPU_UTIL_LOW_WM = 0.5
    NPU_UTIL_HIGH_WM = 98
