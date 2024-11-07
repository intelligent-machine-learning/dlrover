from atorch.utils.fa_util import patch_fa_interface_to_autocast
from atorch.utils.import_util import is_flash_attn_3_avaliable

from ..abstract_extension import AbstractExtension


class FlashAttnFunc3Extension(AbstractExtension):
    def is_available(self) -> bool:
        return is_flash_attn_3_avaliable()

    def load(self):
        if not self.is_available():
            return None, None

        import flash_attn_interface

        patch_fa_interface_to_autocast(flash_attn_interface)

        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa

        return flash_attn_func, flash_attn_varlen_func


flash_attn_func_3, flash_attn_varlen_func_3 = FlashAttnFunc3Extension().load()
