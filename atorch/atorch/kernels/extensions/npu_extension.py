from atorch.utils.import_util import is_torch_npu_available

from .abstract_extension import AbstractExtension


class _NpuExtension(AbstractExtension):
    def is_available(self) -> bool:
        available = False
        try:
            available = is_torch_npu_available()
        except (ImportError, ModuleNotFoundError):
            available = False
        return available
