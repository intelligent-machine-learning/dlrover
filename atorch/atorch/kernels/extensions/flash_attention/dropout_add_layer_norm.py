from typing import Any, Callable, Optional

from ..flash_atten_extension import _FlashAttnExtension


class DropoutAddLayernormExtension(_FlashAttnExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            from flash_attn.ops.layer_norm import dropout_add_layer_norm  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False

        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        from flash_attn.ops.layer_norm import dropout_add_layer_norm  # noqa

        return dropout_add_layer_norm


dropout_add_layer_norm = DropoutAddLayernormExtension().load()
