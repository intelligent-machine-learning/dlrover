from ..torch_xla_extension import _TorchxlaExtension


class FlashAttnXlaExtension(_TorchxlaExtension):
    def is_available(self):
        available = super().is_available()

        if not available:
            return False

        try:
            from torch_xla.core.functions import flash_attn, flash_attn_varlen  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def is_flash_attn_available(self):
        available = super().is_available()

        if not available:
            return False

        try:
            from torch_xla.core.functions import flash_attn  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def is_flash_attn_varlen_available(self):
        available = super().is_available()

        if not available:
            return False

        try:
            from torch_xla.core.functions import flash_attn_varlen  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self):
        if not self.is_available():
            return None, None

        from torch_xla.core.functions import flash_attn, flash_attn_varlen  # noqa

        return flash_attn, flash_attn_varlen

    def load_flash_attn(self):
        if not self.is_flash_attn_available():
            return None

        from torch_xla.core.functions import flash_attn  # noqa

        return flash_attn

    def load_flash_attn_varlen(self):
        if not self.is_flash_attn_varlen_available():
            return None

        from torch_xla.core.functions import flash_attn_varlen  # noqa

        return flash_attn_varlen


_flash_attn_ext = FlashAttnXlaExtension()
xla_flash_attn = _flash_attn_ext.load_flash_attn()
xla_flash_attn_varlen = _flash_attn_ext.load_flash_attn_varlen()
