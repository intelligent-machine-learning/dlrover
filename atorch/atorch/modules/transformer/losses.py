from atorch.utils.import_util import is_triton_available

if is_triton_available():
    from .cross_entropy import AtorchCrossEntropyLoss as CrossEntropyLoss  # noqa F401
else:
    from torch.nn import CrossEntropyLoss  # type:ignore # noqa F401
