"""CUDA-specific tests."""

import pytest


@pytest.mark.cuda
def test_cuda_availability():
    """Test CUDA availability."""
    try:
        import cupy

        assert cupy.cuda.is_available()
    except ImportError:
        pytest.skip("CUDA not available")


@pytest.mark.cuda
def test_gpu_info():
    """Test GPU information retrieval."""
    try:
        import cupy

        device = cupy.cuda.Device()
        assert device.id >= 0
        # Just check that compute_capability exists
        cc = device.compute_capability
        assert cc is not None
    except ImportError:
        pytest.skip("CUDA not available")
