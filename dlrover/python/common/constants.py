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


class PriorityClass(object):
    LOW = "low"
    HIGH = "high"


class PlatformType(object):
    KUBERNETES = "k8s"
    RAY = "ray"
    PY_KUBERNETES = "pyk8s"
    LOCAL = "local"


class ElasticJobApi(object):
    GROUP = "elastic.iml.github.io"
    VERION = "v1alpha1"
    SCALEPLAN_KIND = "ScalePlan"
    SCALEPLAN_PLURAL = "scaleplans"
    ELASTICJOB_KIND = "elasticjob"
    ELASTICJOB_PLURAL = "elasticjobs"


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
    JOB_KEY = "elasticjob.dlrover/name"
    REPLICA_TYPE_KEY = "elasticjob.dlrover/replica-type"
    REPLICA_INDEX_KEY = "elasticjob.dlrover/replica-index"
    RANK_INDEX_KEY = "elasticjob.dlrover/rank-index"


class ScalePlanLabel(object):
    SCALE_TYPE_KEY = "scale-type"
    MANUAL_SCALE = "manual"
    AUTO_SCALE = "auto"


class NodeStatus(object):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    RUNNING = "Running"
    FINISHED = "Finished"
    PENDING = "Pending"
    INITIAL = "Initial"
    DELETED = "Deleted"
    UNKNOWN = "Unknown"
    BREAKDOWN = "Breakdown"


class NodeEventType(object):
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


class NodeExitReason(object):
    KILLED = "Deleted"
    OOM = "OOMKilled"
    FATAL_ERROR = "Error"
    UNKNOWN_ERROR = "UnknownError"
    HARDWARE_ERROR = "HardwareError"


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
    HANG_ERROR = "HangError"


class ExitCode(object):
    FATAL_ERROR_CODE = 1
    KILLED_CODE = 137
    TERMED_CODE = 143
    CORE_DUMP_ERROR_CODE = 134
    OOM_CODE = 247
    GPU_DRIVER_ERROR = 201
    GPU_POD_RESIDUE = 202
    GPU_INFOROM_CORRUPTED = 14


class NodeResourceLimit(object):
    MAX_CPU_CORES = 32
    MIN_CPU_CORES = 4
    MIN_MEMORY = 6144  # 6Gi
    MAX_MEMORY = 65536  # 65536Mi, 64Gi
    MAX_WORKER_NUM = 60
    MAX_PS_NUM = 15
    INCREMENTAL_MEMORY_FACTOR = 2
    MAX_INCREMENTAL_MEMORY = 8192  # 8Gi
    HUGE_MEMORY_THRESHOLD = 102400  # 100Gi
    HUGE_CPU_THRESHOLD = 100
    WAIT_CHIEF_TIMEOUT_SECS = 1800  # 30min
    WAIT_DATA_SHARD_SERVICE_CREATION_SECS = 600  # 10min
    PS_CPU_GROWTH_RATE = 1.2
    PS_CPU_DECREASED_RATE = 0.5
    MIN_VALID_MEMORY = 1024  # 1GB
    MIN_VALID_CPU = 2
    MAX_HANG_TIMEOUT_SECS = 7200  # unit: seconds


class DefaultNodeResource(object):
    PS_NUM = 1
    PS_MEMORY = 8000  # MB
    PS_CPU = 1
    WORKER_NUM = 2
    WORKER_CPU = 1
    WORKER_MEMORY = 8000  # MB


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
    PS = "ParameterServerStrategy"
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
    DLROVER_MASTER_ADDR = "DLROVER_MASTER_ADDR"
    WORKER_TYPE = "WORKER_TYPE"
    WORKER_ID = "WORKER_ID"
    WORKER_NUM = "WORKER_NUM"
    WORKER_RANK = "WORKER_RANK"
    RDZV_ENDPOINT = "RDZV_ENDPOINT"
    GRPC_ENABLE_FORK = "GRPC_ENABLE_FORK_SUPPORT"
    POD_NAME = "POD_NAME"
    AUTO_MONITOR_WORKLOAD = "AUTO_MONITOR_WORKLOAD"
    JOB_NAME = "ELASTIC_JOB_NAME"
    JOB_UID = "JOB_UID"


class DatasetType(object):
    TEXT = "text"
    MAXCOMPUTE_TABLE = "maxcompute_table"


class DefaultResourceLimits(object):
    CPU_LIMIT = 100
    MEMORY_LIMIT = "102400Mi"  # 100Gi
    GPU_LIMIT = 4


class OptimizeMode(object):
    MANNUAL = "manunal"
    SINGLE_JOB = "single-job"
    CLUSTER = "cluster"


class ReporterType(object):
    LOCAL = "local"
    DLROVER_BRAIN = "brain"


class MemoryUnit(object):
    MB = 1024 * 1024


class k8sAPIExceptionReason(object):
    NOT_FOUND = "Not Found"


class RendezvousName(object):
    ELASTIC_TRAINING = "elastic-training"
    NETWORK_CHECK = "network-check"


class NodeErrorMessage(object):
    NETWORKER_ERROR = "Network is breakdown"


class NetworkFailureReason(object):
    NODE_FAILURE = "Node Failure"
    WAITING_NODE = "Waiting node"


class TrainingMsgLevel(object):
    RDZV_ERROR = "rdzv_error"
    PROCESS_ERROR = "process_error"
    NODE_ERROR = "node_error"
    WARNING = "warning"
    INFO = "info"
