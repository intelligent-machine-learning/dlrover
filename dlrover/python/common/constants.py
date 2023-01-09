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


class PlatformType(object):
    KUBERNETES = "k8s"
    PY_KUBERNETES = "pyk8s"


class UserEnv(object):
    USER_ID = "USER_ID"


class TaskType(object):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


class NodeType(object):
    MASTER = "master"
    PS = "ps"
    WORKER = "worker"
    EVALUATOR = "evaluator"
    CHIEF = "chief"
    DLROVER_MASTER = "dlrover-master"


class ElasticJobLabel(object):
    APP_NAME = "dlrover"
    JOB_KEY = "elasticjob-name"
    REPLICA_TYPE_KEY = "replica-type"
    REPLICA_INDEX_KEY = "replica-index"
    RANK_INDEX_KEY = "rank-index"


class NodeStatus(object):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    RUNNING = "Running"
    FINISHED = "Finished"
    PENDING = "Pending"
    INITIAL = "Initial"
    DELETED = "Deleted"


class NodeEventType(object):
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


class NodeExitReason(object):
    KILLED = "Deleted"
    OOM = "OOMKilled"
    FATAL_ERROR = "Error"
    UNKNOWN_ERROR = "UnknownError"


class JobExitReason(object):
    SUCCEEDED = "Completed"
    CODE_ERROR = "CodeError"
    WORKER_OOM = "WorkerOOM"
    WORKER_ERROR = "WorkerError"
    PS_OOM_ERROR = "PSOOM"
    PS_ERROR = "PSError"
    EVALUATOR_OOM = "EvaluatorOOM"
    EVALUATOR_ERROR = "EvaluatorError"
    UNKNOWN_ERROR = "UnknownError"


class ExitCode(object):
    FATAL_ERROR_CODE = 1
    KILLED_CODE = 137
    TERMED_CODE = 143
    CORE_DUMP_ERROR_CODE = 134


class NodeResourceLimit(object):
    MAX_CPU_CORES = 32
    MIN_CPU_CORES = 4
    MIN_MEMORY = 6144  # 6Gi
    MAX_MEMORY = 65536  # 65536Mi, 64Gi
    MAX_WORKER_NUM = 60
    MAX_PS_NUM = 15
    INCREMENTAL_MEMORY_FACTOR = 2
    HUGE_MEMORY_THRESHOLD = 102400  # 100Gi
    HUGE_CPU_THRESHOLD = 100
    WAIT_CHIEF_TIMEOUT_SECS = 1800  # 30min
    WAIT_DATA_SHARD_SERVICE_CREATION_SECS = 600  # 10min
    PS_CPU_GROWTH_RATE = 1.2
    PS_CPU_DECREASED_RATE = 0.5
    MIN_VALID_MEMORY = 1024  # 1GB
    MIN_VALID_CPU = 2


class DefaultNodeResource(object):
    PS_NUM = 3
    PS_MEMORY = 16384  # 16GB
    PS_CPU = 12
    WORKER_NUM = 5
    WORKER_CPU = 16
    WORKER_MEMORY = 16384  # 16GB


class ResourceOptimizerName(object):
    BRAIN = "brain"
    LOCAL = "local"


class JobOptStage(object):
    CREATE = "job_stage_create"
    PS_INITIAL = "job_stage_ps_initial"
    WORKER_INITIAL = "job_stage_worker_initial"
    RUNNING = "job_stage_running"


class OptimizeWorkerPhase(object):
    SAMPLE = "sample"
    INITIAL = "initial"
    STABLE = "stable"


class DistributionStrategy(object):
    LOCAL = "Local"
    PARAMETER_SERVER = "ParameterServerStrategy"
    ALLREDUCE = "AllreduceStrategy"
    CUSTOM = "CustomStrategy"


class PSClusterVersionType(object):
    GLOBAL = "GLOBAL"
    LOCAL = "LOCAL"
    RESTORED = "RESTORED"


class GRPC(object):
    # gRPC limits the size of message by default to 4MB.
    # It's too small to send model parameters.
    MAX_SEND_MESSAGE_LENGTH = 256 * 1024 * 1024
    MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024


class TrainingLoopStatus(object):
    START = 1
    END = 2
    PENDING = 3


class NodeEnv(object):
    RELAUNCHED_POD = "RELAUNCHED_POD"
    ELASTICDL_ENABLED = "ELASTICDL_ENABLED"
    MASTER_ADDR = "DLROVER_MASTER_ADDR"
    WORKER_TYPE = "WORKER_TYPE"
    WORKER_ID = "WORKER_ID"
    WORKER_NUM = "WORKER_NUM"


class DatasetType(object):
    TEXT = "text"
    MAXCOMPUTE_TABLE = "maxcompute_table"


class DefaultResourceLimits(object):
    CPU_LIMIT = 100
    MEMORY_LIMIT = "102400Mi"  # 100Gi
    GPU_LIMIT = 4
