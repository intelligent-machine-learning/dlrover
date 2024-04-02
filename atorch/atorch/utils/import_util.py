import importlib
import os
from functools import lru_cache


def import_module_from_py_file(file_path):
    model_module = None
    if os.path.exists(file_path):
        model_class_path = file_path.replace(".py", "").strip("./")
        model_class_path = model_class_path.replace("/", ".").strip(".")
        model_module = importlib.import_module(model_class_path)
    return model_module


def import_module(module_name):
    func = module_name.split(".")[-1]
    module_path = module_name.replace("." + func, "")
    module = importlib.import_module(module_path)
    return getattr(module, func)


@lru_cache()
def is_torch_npu_available(check_device=False):
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return False
    # Checks if `torch_npu` is installed and potentially if a NPU is in the environment
    if importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_triton_available():
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("triton.language") is None:
        return False
    return True
