import torch

from .abstract_extension import AbstractExtension


class _FlashAttnExtension(AbstractExtension):
    def is_available(self) -> bool:
        available = False
        try:
            import flash_attn  # noqa F401

            available = torch.cuda.is_available()
        except (ImportError, ModuleNotFoundError):
            available = False
        return available
