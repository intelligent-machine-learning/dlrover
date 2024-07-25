from typing import Any, Callable, Optional

from ..npu_extension import _NpuExtension


class FusedAdamwNpuExtension(_NpuExtension):
    def is_available(self) -> bool:

        available = super().is_available()
        if not available:
            return False

        try:
            import torch_npu  # noqa

            available = hasattr(torch_npu, "npu_apply_adam_w")
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        import torch_npu  # noqa

        return torch_npu.npu_apply_adam_w


npu_apply_adam_w = FusedAdamwNpuExtension().load()
