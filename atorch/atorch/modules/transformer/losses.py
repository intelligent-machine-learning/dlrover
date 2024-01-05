try:
    import triton.language as tl  # noqa F401
    from triton import jit  # noqa F401

    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa F401
except (ImportError, ModuleNotFoundError):
    if HAS_TRITON:
        from .cross_entropy import AtorchCrossEntropyLoss as CrossEntropyLoss  # noqa F401
    else:
        from torch.nn import CrossEntropyLoss  # noqa F401
