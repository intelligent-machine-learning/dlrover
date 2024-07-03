from atorch.utils.fa_util import patch_fa_interface_to_autocast

from ..flash_atten_extension import _FlashAttnExtension


class FlashAttnFuncExtension(_FlashAttnExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False

        return available

    def load(self):
        if not self.is_available():
            return None, None

        import flash_attn.flash_attn_interface

        patch_fa_interface_to_autocast(flash_attn.flash_attn_interface)

        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa

        return flash_attn_func, flash_attn_varlen_func


flash_attn_func, flash_attn_varlen_func = FlashAttnFuncExtension().load()
