import torch

from .abstract_extension import AbstractExtension


class _FlashAttn1Extension(AbstractExtension):
    def is_available(self) -> bool:
        available = False
        try:
            import flash_attn_1  # noqa F401

            available = torch.cuda.is_available()
        except (ImportError, ModuleNotFoundError):
            available = False
        return available
