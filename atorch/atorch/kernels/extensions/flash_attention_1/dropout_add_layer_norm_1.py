from typing import Any, Callable, Optional

from ..flash_atten_1_extension import _FlashAttn1Extension


class DropoutAddLayernorm1Extension(_FlashAttn1Extension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            from flash_attn_1.ops.layer_norm import dropout_add_layer_norm  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False

        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        from flash_attn_1.ops.layer_norm import dropout_add_layer_norm  # noqa

        return dropout_add_layer_norm


dropout_add_layer_norm_1 = DropoutAddLayernorm1Extension().load()
