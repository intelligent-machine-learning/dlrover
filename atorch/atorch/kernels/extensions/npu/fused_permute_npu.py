from typing import Any, Callable, Optional

from ..npu_extension import _NpuExtension


class FusedPermuteNpuExtension(_NpuExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            import mindspeed  # noqa

            # Note: compile ops and npu_moe_token_permute in ant mindspeed
            from mindspeed import ops  # noqa
            from mindspeed.ops import npu_moe_token_permute  # noqa

            available = hasattr(mindspeed.ops.npu_moe_token_permute, "npu_moe_token_permute")
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        import mindspeed  # noqa

        # Note: compile ops and npu_moe_token_permute in ant mindspeed
        from mindspeed import ops  # noqa
        from mindspeed.ops import npu_moe_token_permute  # noqa

        return mindspeed.ops.npu_moe_token_permute.npu_moe_token_permute


npu_fused_permute = FusedPermuteNpuExtension().load()
