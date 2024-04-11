try:
    from transformers.utils import is_torch_npu_available
except (ImportError, ModuleNotFoundError):

    def is_torch_npu_available():
        "Checks if `torch_npu` is installed and potentially"
        " if a NPU is in the environment"
        import importlib

        if importlib.util.find_spec("torch_npu") is None:
            return False

        import torch
        import torch_npu  # noqa: F401,F811

        return hasattr(torch, "npu") and torch.npu.is_available()


if is_torch_npu_available():
    import torch_npu  # noqa: F401,F811
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

