import traceback

from atorch.common.log_utils import default_logger as logger

try:
    import deepspeed_npu
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except (ModuleNotFoundError, ImportError):
    logger.error(f"{traceback.format_exc()}")
import torch

old_device_capability = torch.cuda.get_device_capability


def new_device_capability(device):
    if device.type != "cpu":
        return (8, 0)
    else:
        return old_device_capability(device)


def npu_profile_context(*args, **kwargs):
    if "experimental_config" not in kwargs:
        kwargs["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            record_op_args=True,
            data_simplification=False,
        )
    return torch_npu.profiler.profile(*args, **kwargs)


def make_atorch_npu_patch():
    # todo: can't create same device on multiprocessing?
    device = torch.device("npu")
    # # if there is no npu device, there will not make patch
    if torch.npu.get_device_capability(device) is None:
        torch.npu.get_device_capability = new_device_capability
        torch.cuda.get_device_capability = new_device_capability

    torch.profiler.profile = npu_profile_context
    reset_attrs = ["ProfilerActivity", "tensorboard_trace_handler", "schedule"]
    for attr in reset_attrs:
        if hasattr(torch.profiler, attr):
            delattr(torch.profiler, attr)
        setattr(torch.profiler, attr, getattr(torch_npu.profiler, attr))


try:
    make_atorch_npu_patch()
except Exception:
    logger.error(f"{traceback.format_exc()}")
