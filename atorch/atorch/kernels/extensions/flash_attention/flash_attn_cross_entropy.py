from typing import Any, Callable, Optional

from ..flash_atten_extension import _FlashAttnExtension


class FlashAttnCrossEntropyExtension(_FlashAttnExtension):
    def is_available(self) -> bool:
        available = super().is_available()

        if not available:
            return False

        try:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False

        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa

        return CrossEntropyLoss


FlashAttnCrossEntropyLoss = FlashAttnCrossEntropyExtension().load()
