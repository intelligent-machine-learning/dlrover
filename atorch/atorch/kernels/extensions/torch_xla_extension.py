from .abstract_extension import AbstractExtension


class _TorchxlaExtension(AbstractExtension):
    def is_available(self) -> bool:
        available = False
        try:
            import torch_xla  # noqa

            available = True
        except (ImportError, ModuleNotFoundError):
            available = False
        return available
