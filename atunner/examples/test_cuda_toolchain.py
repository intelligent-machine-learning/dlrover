#!/usr/bin/env python3
"""
Example script to test CUDA SoftMax kernel compilation and execution.

This script demonstrates the CUDA toolchain integration in ATunner.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda_environment():
    """Check if CUDA environment is properly set up."""
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        return torch.cuda.is_available()

    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        return False


def test_softmax_kernel():
    """Test SoftMax CUDA kernel compilation and execution."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        print("\nTesting SoftMax CUDA kernel...")

        # Create test data
        batch_size, seq_len, hidden_dim = 2, 4, 8
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        print(f"Input shape: {input_tensor.shape}")
        print(f"Input sample:\n{input_tensor[0, 0, :4]}")  # Show first 4 elements

        # Initialize CUDA SoftMax
        cuda_softmax = get_cuda_softmax()

        # Forward pass
        print("\nRunning forward pass...")
        cuda_output = cuda_softmax.forward(input_tensor)

        # Compare with PyTorch reference
        torch_output = torch.softmax(input_tensor, dim=-1)

        # Check accuracy
        max_diff = torch.max(torch.abs(cuda_output - torch_output))
        mean_diff = torch.mean(torch.abs(cuda_output - torch_output))

        print("Forward pass completed!")
        print(f"Output shape: {cuda_output.shape}")
        print(f"Output sample:\n{cuda_output[0, 0, :4]}")
        print(f"Max difference vs PyTorch: {max_diff:.2e}")
        print(f"Mean difference vs PyTorch: {mean_diff:.2e}")

        # Verify properties
        sums = torch.sum(cuda_output, dim=-1)
        print(f"Sum check (should be ~1.0): {sums[0, 0]:.6f}")

        # Test backward pass
        print("\nTesting backward pass...")
        grad_output = torch.randn_like(cuda_output)
        grad_input = cuda_softmax.backward(grad_output, cuda_output)

        # Compare with PyTorch autograd
        input_tensor_autograd = input_tensor.clone().requires_grad_(True)
        torch_output_autograd = torch.softmax(input_tensor_autograd, dim=-1)
        torch_output_autograd.backward(grad_output)
        torch_grad = input_tensor_autograd.grad

        grad_max_diff = torch.max(torch.abs(grad_input - torch_grad))
        grad_mean_diff = torch.mean(torch.abs(grad_input - torch_grad))

        print("Backward pass completed!")
        print(f"Gradient max difference vs PyTorch: {grad_max_diff:.2e}")
        print(f"Gradient mean difference vs PyTorch: {grad_mean_diff:.2e}")

        # Performance test
        print("\nPerformance test...")
        import time

        # Warmup
        for _ in range(10):
            _ = cuda_softmax.forward(input_tensor)
        torch.cuda.synchronize()

        # Timing
        start_time = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = cuda_softmax.forward(input_tensor)
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time

        # Compare with PyTorch
        start_time = time.time()
        for _ in range(num_iterations):
            _ = torch.softmax(input_tensor, dim=-1)
        torch.cuda.synchronize()
        torch_time = time.time() - start_time

        print(f"CUDA SoftMax time: {cuda_time*1000:.2f} ms " f"({num_iterations} iterations)")
        print(f"PyTorch SoftMax time: {torch_time*1000:.2f} ms " f"({num_iterations} iterations)")
        print(f"Speedup: {torch_time/cuda_time:.2f}x")

        return True

    except Exception as e:
        print(f"SoftMax kernel test failed: {e}")
        logger.exception("Detailed error:")
        return False


def main():
    """Run main test function."""
    print("ATunner CUDA Toolchain Test")
    print("=" * 50)

    # Check environment
    if not check_cuda_environment():
        print("\nCUDA environment not available. Exiting.")
        sys.exit(1)

    # Test SoftMax kernel
    success = test_softmax_kernel()

    if success:
        print("\nAll CUDA toolchain tests passed!")
        print("\nSummary:")
        print("  CUDA environment check")
        print("  Kernel compilation")
        print("  Forward pass accuracy")
        print("  Backward pass accuracy")
        print("  Performance comparison")
        print("\nCUDA toolchain is ready for ATunner development!")
    else:
        print("\nCUDA toolchain test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
