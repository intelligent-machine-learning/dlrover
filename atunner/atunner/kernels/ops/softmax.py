"""
SoftMax operator implementation.

This module provides the Python interface for SoftMax operations,
with CUDA kernel compilation and execution.
"""

import logging

logger = logging.getLogger(__name__)


class SoftMaxOp:
    """SoftMax operation with CUDA acceleration."""

    def __init__(self, cuda_module=None):
        """Initialize SoftMax operator.

        Args:
            cuda_module: Compiled CUDA module containing
                forward/backward functions
        """
        self.cuda_module = cuda_module
        self._compiled = cuda_module is not None

    def forward(self, input_tensor):
        """Forward pass of SoftMax.

        Args:
            input_tensor: Input tensor of shape
                (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor with same shape as input
        """
        if not self._compiled:
            raise RuntimeError(
                "CUDA module not compiled. Use get_cuda_softmax() to " "get a compiled instance."
            )

        # Import torch inside the method to avoid import errors in
        # non-CUDA environments
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required for SoftMax operations")

        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")

        if len(input_tensor.shape) != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, seq, hidden), " f"got shape {input_tensor.shape}"
            )

        return self.cuda_module.forward(input_tensor.contiguous())

    def backward(self, grad_output, softmax_output):
        """Backward pass of SoftMax.

        Args:
            grad_output: Gradient w.r.t. output
            softmax_output: Output from forward pass

        Returns:
            Gradient w.r.t. input
        """
        if not self._compiled:
            raise RuntimeError(
                "CUDA module not compiled. Use get_cuda_softmax() to " "get a compiled instance."
            )

        # Import torch inside the method to avoid import errors in
        # non-CUDA environments
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required for SoftMax operations")

        if not grad_output.is_cuda or not softmax_output.is_cuda:
            raise ValueError("All tensors must be on CUDA device")

        return self.cuda_module.backward(grad_output.contiguous(), softmax_output.contiguous())

    @property
    def is_compiled(self) -> bool:
        """Check if CUDA module is compiled."""
        return self._compiled
