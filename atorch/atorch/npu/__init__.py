import traceback
from typing import Optional, Union

import torch

from atorch.common.log_utils import default_logger as logger
from atorch.utils.import_util import is_torch_npu_available

try:
    if hasattr(torch.device, "__enter__"):
        # NPU bug. Activate DeviceContext before importing torch_npu
        with torch.device("meta"):
            _ = torch.tensor((1.0))
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except (ModuleNotFoundError, ImportError):
    logger.error(f"{traceback.format_exc()}")


_device_t = Union[torch.device, str, int, None]
old_device_capability = torch.cuda.get_device_capability


def new_device_capability(device: Optional[_device_t] = None):
    """

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    Returns:
        tuple(int, int): the major and minor cuda capability of the device
        (8, 0) for npu
    """

    if isinstance(device, (int, str)) or device is None:
        return (8, 0)
    if isinstance(device, torch.device) and device.type != "cpu":
        return (8, 0)
    else:
        return old_device_capability(device)


new_device_capability.__doc__ = old_device_capability.__doc__


def make_atorch_npu_patch():
    # todo: can't create same device on multiprocessing?
    device = torch.device("npu")
    # # if there is no npu device, there will not make patch
    if torch.npu.get_device_capability(device) is None:
        torch.npu.get_device_capability = new_device_capability
        torch.cuda.get_device_capability = new_device_capability

    try:
        import transformers
        from packaging import version

        old_is_torch_bf16_gpu_available = transformers.utils.is_torch_bf16_gpu_available

        def npu_is_torch_bf16_gpu_available():
            if is_torch_npu_available():
                return torch.npu.get_device_name() == "Ascend910B2"
            else:
                return old_is_torch_bf16_gpu_available()

        # transformers does not recognize that 910B support bf16 until 4.35.0
        if version.parse(transformers.__version__) < version.parse("4.35.0"):
            setattr(transformers.utils, "is_torch_bf16_gpu_available", npu_is_torch_bf16_gpu_available)
    except (ModuleNotFoundError, ImportError):
        logger.error(f"{traceback.format_exc()}")


try:
    make_atorch_npu_patch()
except Exception:
    logger.error(f"{traceback.format_exc()}")
