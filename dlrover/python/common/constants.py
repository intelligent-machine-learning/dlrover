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


class BasicClass(object):
    LOG_LEVEL_ENV = "DLROVER_LOG_LEVEL"
    LOG_ROOT_DIR_ENV = "DLROVER_LOG_ROOT_DIR"
    LOG_ROTATE_MAX_BYTES_ENV = "DLROVER_LOG_ROTATE_MAX_BYTES"  # unit: bytes
    LOG_ROTATE_BACKUP_COUNT_ENV = "DLROVER_LOG_ROTATE_BACKUP_COUNT"
    LOG_AGENT_DIR_ENV = "DLROVER_LOG_AGENT_DIR"


class PriorityClass(object):
    LOW = "low"
    HIGH = "high"


class PlatformType(object):
    KUBERNETES = "k8s"
    RAY = "ray"
    PY_KUBERNETES = "pyk8s"
    LOCAL = "local"


class CommunicationType(object):
    COMM_SERVICE_GRPC = "grpc"
    COMM_SERVICE_HTTP = "http"
    COMM_SERVICE_RAY = "ray"


class CommunicationReqType(object):
    COMM_REQ_TYPE_GET = "get"
    COMM_REQ_TYPE_REPORT = "report"


class CommunicationReqMeta(object):
    COMM_META_JOB_UID = "job_uid"
    COMM_META_JOB_UID_INVALID_MSG = "Job uid is invalid"


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
    RELAUNCH_COUNT = "elasticjob.dlrover/relaunch-count"


class SchedulingLabel(object):
    NODE_GROUP = "scheduling/rack-group"  # int, the node group index
    NODE_GROUP_SIZE = (
        "scheduling/rack-group-num"  # int, the total node group numbers
    )
    NODE_GROUP_ID = "scheduling/rack-id"  # str, node group name


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

    @classmethod
    def is_terminal_status(cls, status):
        if status in [
            NodeStatus.FAILED,
            NodeStatus.DELETED,
            NodeStatus.FINISHED,
            NodeStatus.SUCCEEDED,
        ]:
            return True
        return False


class NodeEventType(object):
    """Notice: the type here is equal to the pod event type by k8s"""

    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    ERROR = "ERROR"
    FINISHED = "FINISHED"
    SUCCEEDED_EXITED = "SUCCEEDED_EXITED"
    FAILED_EXITED = "FAILED_EXITED"
    NODE_CHECK_SUCCEEDED = "NODE_CHECK_SUCCEEDED"
    NODE_CHECK_FAILED = "NODE_CHECK_FAILED"
    MASTER_CONNECTION_FAILED = "MASTER_CONNECTION_FAILED"
    WAIT_PRE_CHECK = "WAIT_PRE_CHECK"


class PendingTimeoutStrategyType(object):
    SKIP = 0
    NECESSARY = 1
    ALL = 2


class GpuMetricEnum(object):
    """
    it is the metrics enum of nvidia GPU, collected by DCGM
    """

    GPU_FREE_MEM = "DCGM_FI_DEV_FB_FREE"
    GPU_USED_MEM = "DCGM_FI_DEV_FB_USED"
    GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
    GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
    GPU_SM_UTIL = "DCGM_FI_PROF_SM_OCCUPANCY"
    GPU_SM_ACTIVE = "DCGM_FI_PROF_SM_ACTIVE"
    GPU_TENSOR_UTIL = "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"


class NpuMetricEnum(object):
    """
    it is the metrics enum of Ascend NPU, collected by DCMI
    """

    NPU_TOTAL_MEM = "npu_chip_info_total_memory"
    NPU_USED_MEM = "npu_chip_info_used_memory"
    NPU_UTIL = "npu_chip_info_utilization"
    NPU_TEMP = "npu_chip_info_temperature"
    NPU_HEALTH_STATE = "npu_chip_info_health_status"
    NPU_LINK_STATE = "npu_chip_info_link_status"
    NPU_OPTICAL_STATE = "npu_chip_optical_state"
    NPU_NETWORK_STATE = "npu_chip_info_network_status"
    NPU_RDMA_RX = "npu_chip_info_bandwidth_rx"
    NPU_RDMA_TX = "npu_chip_info_bandwidth_tx"


class MthreadsGPUMetricEnum(object):
    """
    it is the metrics enum of Mthreads GPU, collected by Mthreads
    """

    GPU_FREE_MEM = "DCGM_FI_DEV_FB_FREE"
    GPU_USED_MEM = "DCGM_FI_DEV_FB_USED"
    GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
    GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
    GPU_SM_UTIL = "DCGM_FI_PROF_MP_OCCUPANCY"
    GPU_TENSOR_UTIL = "DCGM_FI_PROF_PIPE_MT_TENSOR_ACTIVE"


class NodeExitReason(object):
    Succeeded = "Succeeded"
    KILLED = "Deleted"
    OOM = "OOMKilled"
    FATAL_ERROR = "Error"
    UNKNOWN_ERROR = "UnknownError"
    HARDWARE_ERROR = "HardwareError"
    NO_HEARTBEAT = "NoHeartBeat"
    DIAG_FAIL = "DiagnosticFailure"
    RELAUNCHED = "Relaunched"
    CHECK_FAIL = "CheckFailure"


class NodeExitDescription(object):
    GPU_INVALID_MSG = "The node's GPU basic check failed."
    NODE_STRAGGLE_MSG = "The node is a straggler and exits."
    NODE_FAILED_MSG = "This node failed the node-check procedure(mat-mul + comm) before training."


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
    RDZV_TIMEOUT_ERROR = "RdzvTimeout"
    PENDING_TIMEOUT = "PendingTimeout"
    UNCOMPLETED_TIMEOUT = "UncompletedTimeout"
    NODE_CHECK_FAILED = "NodeCheckFailed"
    BY_INTERNAL_SUGGESTION = "ByInternalSuggestion"
    RESTART_OVER_LIMIT = "RestartOverLimit"


class CustomMetricKeys:
    RDZV_ROUND = "rdzv_round"
    TRAINING_ERROR_LEVEL = "error_level"
    ERROR_CONTENT = "error_content"


class ExitCode(object):
    FATAL_ERROR_CODE = 1
    KILLED_CODE = 137
    TERMED_CODE = 143
    CORE_DUMP_ERROR_CODE = 134
    OOM_CODE = 247
    GPU_DRIVER_ERROR = 201
    GPU_POD_RESIDUE = 202
    GPU_INFOROM_CORRUPTED = 14
    CONTAINER_FAILED_OR_UNKNOWN_DEVICE = 128


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
    PS_RUNNING = "job_stage_ps_running"
    WORKER_RUNNING = "job_stage_worker_running"


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
    DLROVER_MASTER_SERVICE_TYPE = "DLROVER_MASTER_SERVICE_TYPE"
    DLROVER_TRAINING_ELASTIC_MODE = "DLROVER_TRAINING_ELASTIC_MODE"
    GRPC_ENABLE_FORK = "GRPC_ENABLE_FORK_SUPPORT"
    GRPC_POLL_STRATEGY = "GRPC_POLL_STRATEGY"
    POD_NAME = "POD_NAME"
    MONITOR_ENABLED = "MONITOR_ENABLED"
    JOB_NAME = "ELASTIC_JOB_NAME"
    JOB_UID = "JOB_UID"

    NODE_TYPE = "NODE_TYPE"
    NODE_ID = "NODE_ID"
    NODE_NUM = "NODE_NUM"
    NODE_RANK = "NODE_RANK"

    # Deprecated env vars.
    WORKER_TYPE = "WORKER_TYPE"
    WORKER_ID = "WORKER_ID"
    WORKER_NUM = "WORKER_NUM"
    WORKER_RANK = "WORKER_RANK"

    # The envs are compatibile with kubeflow/PytorchJob.
    RANK = "RANK"  # It is the rank of node not the rank of process.
    WORLD_SIZE = "WORLD_SIZE"  # It is the number of nodes.

    # worker process env
    TORCHELASTIC_RUN_ID = "TORCHELASTIC_RUN_ID"
    MASTER_ADDR = "MASTER_ADDR"
    MASTER_PORT = "MASTER_PORT"

    # diagnosis env
    TRAINING_LOG_FILE = "TRAINING_LOG_FILE"
    FAILURE_NODE_ERRORS = "FAILURE_NODE_ERRORS"

    # grpc env
    MASTER_CLIENT_TIMEOUT = "MASTER_CLIENT_TIMEOUT"

    # extension env
    DLROVER_EXTENSION_DYNAMIC_FAILOVER = "DLROVER_EXTENSION_DYNAMIC_FAILOVER"


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
    TRAINING = "elastic-training"
    NETWORK_CHECK = "network-check"


class RendezvousErrorType(object):
    PEND_TIMEOUT = "pend-timeout"
    JOIN_TIMEOUT = "join-timeout"


class NetworkFailureReason(object):
    NO_INIT = "Not Initialized"
    NODE_FAILURE = "Node Failure"
    WAITING_NODE = "Waiting node"


class TrainingExceptionLevel(object):
    RDZV_ERROR = "rdzv_error"
    PROCESS_ERROR = "process_error"
    NODE_ERROR = "node_error"
    WARNING = "warning"
    INFO = "info"
    ERROR = "error"


class ConfigPath(object):
    ENV_PARAL_CONFIG = "DLROVER_PARAL_CONFIG_PATH"
    PARAL_CONFIG = "/tmp/dlrover/auto_paral_config.json"
    ENV_RUNTIME_METRICS = "RUNTIME_METRICS_PATH"
    RUNTIME_METRICS = "/tmp/dlrover/runtime_metrics.json"
    NETWORK_CHECK_DATA_DIR = "/tmp/dlrover/network_check/"


class ScriptPath(object):
    RUN_AFFINITY_SCRIPT = "dlrover_run_affinity.sh"


class CheckpointConstant(object):
    TRACER_FILE_NAME = "dlrover_latest.txt"
    MODEL_STATES_NAME = "model_states"
    OPTIM_STATES_NAME = "optim_states"
    SAVE_TIMEOUT = 600


class JobConstant(object):
    RDZV_JOIN_TIMEOUT_DEFAULT = 600
    INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MIN = 600
    INSUFFICIENT_NODE_TIMEOUT_DEFAULT_MAX = 3600
    PENDING_NODE_TIMEOUT_DEFAULT_MIN = 600
    GROUP_PENDING_NODE_TIMEOUT_DEFAULT_MIN = 300
    NODE_CHECK_TIMEOUT = 300
    SUCCEEDED_POD_TERMINATING_TIMEOUT = 1200

    # timeout 60s
    MASTER_CLIENT_DEFAULT_TIMEOUT = 60

    # grpc timeout 60s
    MASTER_CLIENT_GRPC_DEFAULT_TIMEOUT = 60

    # master_client.check_fault_node/check_straggler timeout value
    # must > NODE_CHECK_TIMEOUT
    MASTER_CLIENT_CHECK_NODE_TIMEOUT = 360

    # sleep 1s on NetworkFailureReason.WAITING_NODE
    MASTER_CLIENT_CHECK_FAULT_SLEEP_TIMEOUT = 1

    # sleep 1s on NetworkFailureReason.WAITING_NODE
    MASTER_CLIENT_CHECK_STRAGGLER_SLEEP_TIMEOUT = 1

    # sleep 5s before next node check round
    NODE_CHECK_NEXT_ROUND_TIMEOUT = 5

    # default interval seconds for loop in training agent
    TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15

    # sleep 5s before next rendezvous round
    RENDEZVOUS_DEFAULT_INTERVAL = 5

    # sleep 5s before next port synchronization
    SYNC_PORTS_DEFAULT_INTERVAL = 5

    # interval seconds for pre-check waiting
    PRE_CHECK_WAIT_SECS = 10


class Accelerators(object):
    NVIDIA_GPU = "nvidia.com/gpu"
    ASCEND_NPU = "ascend-npu"
    MTHREADS_GPU = "mthreads.com/gpu"
    GENERIC_CPU = "cpu"


class AscendConstants(object):
    # By default， there are 16(max) npu on one machine
    NPU_PER_NODE = 16

    # represent the starting offset of the hccl's port using
    HCCL_PORT_START = "HCCL_IF_BASE_PORT"
    HCCL_PORT_START_DEFAULT = 64000


class EventReportConstants(object):
    TYPE_INFO = "info"
    TYPE_WARN = "warn"
    TYPE_ERROR = "error"

    JOB_INSTANCE = "job"

    ACTION_MASTER_START = "master_start"
    ACTION_MASTER_END = "master_end"
    ACTION_JOB_START = "job_start"
    ACTION_JOB_SUCCESS = "job_success"
    ACTION_JOB_FAIL = "job_fail"
    ACTION_WORKER_CREATE = "worker_create"
    ACTION_WORKER_PENDING = "worker_pending"
    ACTION_WORKER_NO_HEARTBEAT = "worker_no_heartbeat"
    ACTION_STATUS_UPDATE = "status_update"
    ACTION_EARLY_STOP = "early_stop"
    ACTION_STOP = "stop"
    ACTION_RESTART = "restart"
    ACTION_RELAUNCH = "relaunch_worker"
    ACTION_NOT_RELAUNCH = "not_relaunch_worker"
    ACTION_GLOBAL_STEP = "global_step"
    ACTION_RDZV_JOIN = "join_rendezvous"
    ACTION_RDZV_COMPLETE = "rendezvous_complete"
    ACTION_RDZV_TIMEOUT = "rendezvous_timeout"
    ACTION_TRAINING_START = "training_start"
    ACTION_RESTART_TRAINING = "restart_training"
    ACTION_SAVE_SHARD_START = "save_shard_start"
    ACTION_SAVE_SHARD_COMPLETE = "save_shard_complete"
    ACTION_SAVE_SHARD_ERROR = "save_shard_error"
    ACTION_MEM_CKPT_START = "mem_ckpt_start"
    ACTION_MEM_CKPT_COMPLETE = "mem_ckpt_complete"
    ACTION_RESUME_MEM_CKPT_START = "resume_mem_ckpt_start"
    ACTION_RESUME_MEM_CKPT_COMPLETE = "resume_mem_ckpt_complete"
    ACTION_HANG_WARN = "hang_warning"
    ACTION_DEVICE_WARNING = "device_warning"

    ACTION_PRE_CHECK_DISABLE = "pre_check_disable"
    ACTION_PRE_CHECK_TIMEOUT = "pre_check_timeout"
    ACTION_PRE_CHECK_ERROR = "pre_check_error"
    ACTION_PRE_CHECK_PASS = "pre_check_pass"
    ACTION_PRE_CHECK_FAIL = "pre_check_fail"


class PreCheckStatus(object):
    CHECKING = "CHECKING"
    FAIL = "FAIL"
    PASS = "PASS"
    DISABLED = "DISABLED"


class DictKey(object):
    """
    Presents the keys in dictionary
    """

    # the key of logs
    LOGS = "logs"


class JobStage(object):
    JOB_INIT = "init"
    JOB_RUNNING = "running"
    JOB_STOPPING = "stopping"
    JOB_STOPPED = "stopped"
    JOB_SUSPENDED = "suspended"
    JOB_RESTARTING = "restarting"


class KeyValueOps(object):
    ADD = "add"
    APPEND = "append"
    GET = "get"
    SET = "set"
    DELETE = "delete"
