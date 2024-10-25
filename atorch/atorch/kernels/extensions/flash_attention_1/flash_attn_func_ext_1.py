from typing import Any, Callable, Optional

from atorch.utils.fa_util import patch_fa_interface_to_autocast

from ..flash_atten_1_extension import _FlashAttn1Extension


class FlashAttnFunc1Extension(_FlashAttn1Extension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            from flash_attn_1.flash_attn_interface import flash_attn_unpadded_func  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False

        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        import flash_attn_1.flash_attn_interface  # noqa

        patch_fa_interface_to_autocast(flash_attn_1.flash_attn_interface)

        from flash_attn_1.flash_attn_interface import flash_attn_unpadded_func  # noqa

        return flash_attn_unpadded_func


flash_attn_unpadded_func_1 = FlashAttnFunc1Extension().load()
