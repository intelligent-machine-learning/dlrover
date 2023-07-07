from enum import Enum


class GrpcEnv(object):
    """
    The default message size of grpc is 4 MB, which is too small to send
    training data, model parameters, or other type of data.
    """

    MAX_SEND_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1 GB
    MAX_RECEIVE_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1 GB


class AnalyserConstants(object):
    MODEL_PARAMS_NUM = "model_params_num"
    MODEL_PARAMS_MB = "model_params_mb"
    MODEL_FLOPS_AND_DYNAMIC_MEMORY_MB = "model_flops_and_dynamic_memory_mb"
    HAS_MODULE_FOR_REPLACE = "has_module_for_replace"
    OPTIMIZER_TYPE = "optimizer_info"
    ANALYSE_BASIC = "analyse_basic"
    ANALYSE_TRANSFORMER = "analyse_transformer"
    ANALYSE_DYNAMIC = "analyse_dynamic"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    DATA_SIZE = "data_size"
    FIXED_DATA_SIZE = "fixed_data_size"
    OPTIMIZER_STATE_NUM_AND_MEMORY_MB = "optimizer_state_num_and_memory_mb"
    OPTIMIZER_STATE_NUM = "optimizer_state_num"
    OPTIMIZER_STATE_MEMORY_MB = "optimizer_state_memory_mb"
    MODEL_FLOPS = "model_flops"
    DYNAMIC_MEMORY_MB = "dynamic_memory_mb"
    TRANSFORMER_SEQUENCE_LENGTH = "transformer_sequence_length"
    SUBMODULE_TYPES = "submodule_types"
    OPT_CONFIG_SUBMODULE_NAMES = "opt_config_submodule_names"
    PROFILE_DIR = "/home/admin/profile"
    PROF_FILE_NAME = "aprof.txt"
    TIMELINE_FILE_NAME = "aprof.json"
    TIMELINE_SIGNAL_FILE_NAME = "aprof.done"
    PROF_SIGNAL_FILE_NAME = "aprof_txt.done"
    GPU_UTILIZATION = "gpu_utility"


class AutoAccelerateExtraArgs(Enum):
    FIND_UNUSED_PARAMETERS = "find_unused_parameters"
    SAMPLE_BATCH = "sample_batch"
    BATCH_SIZE = "batch_size"  # total batch size. Equals to batch_size_per_process * ddp_size
    EXPAND_SAMPLE_BATCH = "expand_sample_batch"  # whether to expand sample batch

    @classmethod
    def all(cls):
        return [variable.value for variable in list(cls)]


class GPUCapability:
    """
    TFLOPS of GPU
    """

    TFLOPS = {
        "FP16": {"NVIDIA A100-SXM4-80GB": 312, "Tesla V100-SXM2-32GB": 125},
        "FP32": {"NVIDIA A100-SXM4-80GB": 19.5, "Tesla V100-SXM2-32GB": 15.7},
    }
