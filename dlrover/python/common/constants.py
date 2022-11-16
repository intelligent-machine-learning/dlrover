# Copyright 2022 The DLRover Authors. All rights reserved.
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


class UserEnv(object):
    USER_ID = "USER_ID"


class TaskType(object):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


class NodeType(object):
    MASTER = "edl-master"
    PS = "ps"
    WORKER = "worker"
    EVALUATOR = "evaluator"
    TF_MASTER = "master"



class ElasticJobLabel(object):
    APP_NAME = "elasticdl"
    JOB_KEY = "elasticdl-job-name"
    REPLICA_TYPE_KEY = "replica-type"
    REPLICA_INDEX_KEY = "replica-index"
    TRAINING_TASK_INDEX_KEY = "training-task-index"


class NodeStatus(object):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    RUNNING = "Running"
    FINISHED = "Finished"
    PENDING = "Pending"
    INITIAL = "Initial"
    DELETED = "Deleted"


class NodeExitReason(object):
    KILLED = "Deleted"
    OOM = "OOMKilled"
    FATAL_ERROR = "Error"
    UNKNOWN_ERROR = "UnknownError"


class ExitCode(object):
    FATAL_ERROR_CODE = 1
    KILLED_CODE = 137
    TERMED_CODE = 143
    CORE_DUMP_ERROR_CODE = 134


class NodeResourceBoundary(object):
    MAX_CPU_CORES = 32
    MIN_CPU_CORES = 4
    MIN_MEMORY = 6144  # 6Gi
    MAX_MEMORY = 65536  # 65536Mi, 64Gi
    MAX_WORKER_NUM = 60
    MAX_PS_NUM = 15
    INCREMENTAL_MEMORY_FACTOR = 2
    HUGE_MEMORY_THRESHOLD = 102400  # 100Gi
    HUGE_CPU_THRESHOLD = 100
    WAIT_CHIEF_WORKER_TIMEOUT_SECS = 1800  # 30min
    WAIT_DATA_SHARD_SERVICE_CREATION_SECS = 600  # 10min
    PS_CPU_GROWTH_RATE = 1.2
    PS_CPU_DECREASED_RATE = 0.5
    MIN_VALID_MEMORY = 1024  # 1GB
    MIN_VALID_CPU = 2
