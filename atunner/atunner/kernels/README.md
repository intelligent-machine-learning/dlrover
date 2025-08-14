# ATunner CUDA Kernels

Custom CUDA kernel implementations for high-performance deep learning operations.

## Architecture

The kernels package is organized into three main components:

### Directory Structure

```
atunner/kernels/
├── ops/                    # Python operator implementations
│   ├── __init__.py
│   └── softmax.py         # SoftMax operator class
├── cc/                     # CUDA C++ source files
│   ├── __init__.py
│   └── softmax_cuda.cu    # SoftMax CUDA kernel source
├── compiler.py            # CUDA kernel compilation utilities
├── __init__.py            # Package interface
└── README.md              # This file
```

### Components

- **ops/**: Contains Python operator classes that provide high-level interfaces to CUDA kernels
- **cc/**: Contains CUDA C++ source files (.cu) with kernel implementations
- **compiler.py**: Handles just-in-time compilation of CUDA kernels using PyTorch's extension system

## Current Implementation

### SoftMax Kernel

High-performance CUDA implementation of the SoftMax operation with:
- Numerically stable computation using max reduction
- Optimized memory access patterns
- Support for arbitrary input shapes (batch_size, seq_len, hidden_dim)
- Forward and backward pass implementations

**Files:**
- `ops/softmax.py`: Python SoftMaxOp class
- `cc/softmax_cuda.cu`: CUDA kernel implementation

## Usage

### Basic SoftMax Example

```python
import torch
from atunner.kernels import get_cuda_softmax

# Create input tensor
input_tensor = torch.randn(2, 4, 8, device='cuda')

# Get CUDA SoftMax operator
cuda_softmax = get_cuda_softmax()

# Forward pass
output = cuda_softmax.forward(input_tensor)

# Backward pass
grad_output = torch.randn_like(output)
grad_input = cuda_softmax.backward(grad_output, output)
```

### Advanced Usage

```python
from atunner.kernels import get_cuda_compiler
from atunner.kernels.ops import SoftMaxOp

# Compile kernel manually
compiler = get_cuda_compiler()
cuda_module = compiler.compile_from_file("softmax_cuda.cu", "my_softmax")

# Create operator with custom module
softmax_op = SoftMaxOp(cuda_module)
```

## Testing

Run CUDA-specific tests:
```bash
# Basic CUDA environment tests
make test-cuda

# SoftMax kernel tests (requires compilation)
pytest tests/test_cuda/test_softmax.py -v -m cuda

# Quick toolchain verification
python examples/test_cuda_toolchain.py
```

## Environment Setup

For optimal compilation performance:
```bash
# Install CUDA dependencies
pip install -e .[cuda]

# Set GPU architecture (optional, speeds up compilation)
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
export TORCH_CUDA_ARCH_LIST="7.0"  # For V100
```

## Architecture Support

- **sm_70**: Tesla V100, Titan V
- **sm_80**: A100, A40
- **sm_89**: H100 (future support)

## Performance Features

- **JIT Compilation**: Dynamic kernel compilation
- **Multi-GPU**: Support for different architectures
- **Caching**: PyTorch extension caching
- **Optimization**: -O3 compilation flags

## Future Enhancements

- [ ] Template-based kernel generation
- [ ] Kernel fusion optimization
- [ ] Performance profiling integration
