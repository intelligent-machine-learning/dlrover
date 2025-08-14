"""Test cases for CUDA SoftMax kernel implementation."""

import pytest


@pytest.mark.cuda
def test_cuda_available():
    """Test if CUDA is available for testing."""
    try:
        import torch

        assert torch.cuda.is_available(), "CUDA not available for testing"
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.mark.cuda
def test_softmax_compilation():
    """Test SoftMax CUDA kernel compilation."""
    try:
        from atunner.kernels import get_cuda_softmax

        cuda_softmax = get_cuda_softmax()
        assert cuda_softmax is not None
        print("SoftMax CUDA kernel compilation test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"CUDA kernel compilation failed: {e}")


@pytest.mark.cuda
def test_softmax_forward_basic():
    """Test basic SoftMax forward pass."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        # Create test input
        batch_size, seq_len, hidden_dim = 2, 4, 8
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        # Test CUDA implementation
        cuda_softmax = get_cuda_softmax()
        cuda_output = cuda_softmax.forward(input_tensor)

        # Compare with PyTorch reference
        torch_output = torch.softmax(input_tensor, dim=-1)

        # Check shapes match
        assert (
            cuda_output.shape == torch_output.shape
        ), f"Shape mismatch: {cuda_output.shape} vs {torch_output.shape}"

        # Check numerical accuracy
        max_diff = torch.max(torch.abs(cuda_output - torch_output))
        assert max_diff < 1e-5, f"Max difference too large: {max_diff}"

        print("SoftMax forward pass test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"SoftMax forward test failed: {e}")


@pytest.mark.cuda
def test_softmax_backward():
    """Test SoftMax backward pass."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        batch_size, seq_len, hidden_dim = 1, 3, 5
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True
        )

        cuda_softmax = get_cuda_softmax()
        output = cuda_softmax.forward(input_tensor)

        # Create dummy gradient
        grad_output = torch.ones_like(output)
        grad_input = cuda_softmax.backward(grad_output, output)

        # Test shape consistency
        assert grad_input.shape == input_tensor.shape, (
            f"Gradient shape mismatch: {grad_input.shape} vs " f"{input_tensor.shape}"
        )

        # Test gradient is finite
        assert torch.all(torch.isfinite(grad_input)), "Gradient should be finite"

        print("SoftMax backward pass test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"SoftMax backward test failed: {e}")


@pytest.mark.cuda
def test_softmax_properties():
    """Test SoftMax mathematical properties."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        batch_size, seq_len, hidden_dim = 1, 3, 5
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        cuda_softmax = get_cuda_softmax()
        output = cuda_softmax.forward(input_tensor)

        # Test: All outputs should be positive
        assert torch.all(output >= 0), "SoftMax output should be non-negative"

        # Test: Sum along last dimension should be 1
        sums = torch.sum(output, dim=-1)
        expected_sums = torch.ones_like(sums)
        assert torch.allclose(
            sums, expected_sums, atol=1e-6
        ), f"SoftMax sums should be 1, got {sums}"

        print("SoftMax properties test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"SoftMax properties test failed: {e}")


@pytest.mark.cuda
def test_softmax_numerical_stability():
    """Test SoftMax numerical stability with large inputs."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        # Create input with large values that could cause overflow
        input_tensor = torch.tensor(
            [[[100.0, 101.0, 99.0, 102.0], [200.0, 201.0, 199.0, 202.0]]],
            device="cuda",
        )

        cuda_softmax = get_cuda_softmax()
        output = cuda_softmax.forward(input_tensor)

        # Check that output is still valid (no NaN/Inf)
        assert torch.all(torch.isfinite(output)), "Output should be finite"
        assert not torch.any(torch.isnan(output)), "Output should not contain NaN"

        # Check that probabilities sum to 1
        sums = torch.sum(output, dim=-1)
        assert torch.allclose(
            sums, torch.ones_like(sums), atol=1e-6
        ), "Probabilities should sum to 1 even with large inputs"

        print("SoftMax numerical stability test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"SoftMax numerical stability test failed: {e}")


@pytest.mark.cuda
def test_softmax_different_shapes():
    """Test SoftMax with different tensor shapes."""
    try:
        import torch

        from atunner.kernels import get_cuda_softmax

        cuda_softmax = get_cuda_softmax()

        test_shapes = [
            (1, 1, 1),  # Minimal case
            (1, 1, 10),  # Single sequence, multiple features
            (1, 5, 1),  # Multiple sequences, single feature
            (3, 7, 12),  # Medium size
        ]

        for batch_size, seq_len, hidden_dim in test_shapes:
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
            output = cuda_softmax.forward(input_tensor)

            # Check shape preservation
            assert (
                output.shape == input_tensor.shape
            ), f"Shape not preserved for {input_tensor.shape}"

            # Check probability properties
            sums = torch.sum(output, dim=-1)
            expected_sums = torch.ones(batch_size, seq_len, device="cuda")
            assert torch.allclose(
                sums, expected_sums, atol=1e-6
            ), f"Failed for shape {input_tensor.shape}"

        print("SoftMax different shapes test passed")

    except ImportError as e:
        pytest.skip(f"Missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"SoftMax shapes test failed: {e}")
