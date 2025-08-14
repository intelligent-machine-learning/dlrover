"""Test configuration file."""

import sys
from pathlib import Path

import pytest

# Add the atunner package to the Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_operator_definition():
    """Sample operator definition for testing."""
    return "conv2d"


@pytest.fixture
def sample_input_tensors():
    """Sample input tensor information for testing."""
    from atunner.core.base import TensorInfo

    return [
        TensorInfo(
            shape=[1, 3, 224, 224],
            dtype="float32",
            device="cuda",
            layout="NCHW",
        )
    ]


@pytest.fixture
def sample_hardware_spec():
    """Sample hardware specification for testing."""
    from atunner.core.base import HardwareSpec

    return HardwareSpec(
        gpu_model="A100",
        compute_capability="8.0",
        memory_size="40GB",
        memory_bandwidth="1555GB/s",
        sm_count=108,
        tensor_cores=True,
    )
