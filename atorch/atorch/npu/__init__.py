from atorch.common.log_utils import default_logger as logger

try:
    import deepspeed_npu
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except (ModuleNotFoundError, ImportError) as e:
    logger.error(f"{e}")
    pass
