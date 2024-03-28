from atorch.utils.import_util import is_triton_available

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa F401
except (ImportError, ModuleNotFoundError):
    if is_triton_available():
        from .cross_entropy import AtorchCrossEntropyLoss as CrossEntropyLoss  # noqa F401
    else:
        from torch.nn import CrossEntropyLoss  # noqa F401
