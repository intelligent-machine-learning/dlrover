from typing import Any, Callable, Optional

from ..npu_extension import _NpuExtension


class FlashAttentionNpuExtension(_NpuExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            import torch_npu  # noqa

            available = hasattr(torch_npu, "npu_fusion_attention")
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        import torch_npu  # noqa

        return torch_npu.npu_fusion_attention


npu_fusion_attention = FlashAttentionNpuExtension().load()
