"""
ATunner CUDA Kernels.

This package provides CUDA kernel implementations and compilation utilities
for high-performance deep learning operations.

Structure:
- ops/: Python operator implementations
- cc/: CUDA C++ source files
- compiler.py: CUDA kernel compilation utilities
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import compilation utilities
try:
    from .compiler import compile_softmax_kernel

    _COMPILER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CUDA compiler not available: {e}")
    _COMPILER_AVAILABLE = False

# Import operators
try:
    from .ops import SoftMaxOp

    _OPS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Operators not available: {e}")
    _OPS_AVAILABLE = False


def get_cuda_softmax() -> Optional[Any]:
    """Get compiled CUDA SoftMax operator.

    Returns:
        Compiled SoftMax operator instance, or None if compilation fails
    """
    if not _COMPILER_AVAILABLE or not _OPS_AVAILABLE:
        logger.error("CUDA compiler or operators not available")
        return None

    try:
        # Compile CUDA kernel
        cuda_module = compile_softmax_kernel()

        # Create operator instance
        softmax_op = SoftMaxOp(cuda_module)

        return softmax_op

    except Exception as e:
        logger.error(f"Failed to compile SoftMax CUDA kernel: {e}")
        return None


# Backward compatibility: maintain CUDASoftMax class interface
class CUDASoftMax:
    """Backward compatibility wrapper for SoftMax operator."""

    def __init__(self):
        self._op = get_cuda_softmax()
        if self._op is None:
            raise RuntimeError("Failed to initialize CUDA SoftMax operator")

    def forward(self, input_tensor):
        """Forward pass."""
        return self._op.forward(input_tensor)

    def backward(self, grad_output, softmax_output):
        """Backward pass."""
        return self._op.backward(grad_output, softmax_output)


# Export public API
__all__ = []

if _COMPILER_AVAILABLE:
    __all__.extend(["CUDAKernelCompiler", "get_cuda_compiler", "compile_softmax_kernel"])

if _OPS_AVAILABLE:
    __all__.extend(["SoftMaxOp"])

if _COMPILER_AVAILABLE and _OPS_AVAILABLE:
    __all__.extend(["get_cuda_softmax", "CUDASoftMax"])
