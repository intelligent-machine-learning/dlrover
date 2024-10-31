from typing import Any, Callable, Optional

from ..npu_extension import _NpuExtension


class FusedCrossEntropyNpuExtension(_NpuExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            import mindspeed  # noqa

            # Note: compile ops and npu_fuse_cross_entropy_loss in ant mindspeed
            from mindspeed import ops  # noqa
            from mindspeed.ops import npu_fuse_cross_entropy_loss  # noqa

            available = hasattr(mindspeed.ops.npu_fuse_cross_entropy_loss, "npu_fuse_cross_entropy_loss")
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        import mindspeed  # noqa

        # Note: compile ops and npu_fuse_cross_entropy_loss in ant mindspeed
        from mindspeed import ops  # noqa
        from mindspeed.ops import npu_fuse_cross_entropy_loss  # noqa

        return mindspeed.ops.npu_fuse_cross_entropy_loss.npu_fuse_cross_entropy_loss


npu_fuse_cross_entropy_loss = FusedCrossEntropyNpuExtension().load()
