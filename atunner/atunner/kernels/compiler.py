"""
CUDA kernel compiler for ATunner.

This module handles compilation of CUDA kernels using
torch.utils.cpp_extension.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CUDAKernelCompiler:
    """Compiler for CUDA kernels."""

    def __init__(self):
        """Initialize the CUDA kernel compiler."""
        self._compiled_modules: Dict[str, Any] = {}
        self._kernels_dir = Path(__file__).parent
        self._cc_dir = self._kernels_dir / "cc"

    def compile_from_source(
        self,
        source_code: str,
        name: str,
        extra_cflags=None,
        extra_cuda_cflags=None,
    ) -> Any:
        """Compile CUDA kernel from source code string.

        Args:
            source_code: CUDA C++ source code
            name: Name for the compiled module
            extra_cflags: Additional C++ compiler flags
            extra_cuda_cflags: Additional CUDA compiler flags

        Returns:
            Compiled CUDA module
        """
        if name in self._compiled_modules:
            return self._compiled_modules[name]

        try:
            import torch
            from torch.utils.cpp_extension import load_inline
        except ImportError:
            raise ImportError("PyTorch is required for CUDA kernel compilation")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Detect GPU architecture and set TORCH_CUDA_ARCH_LIST to avoid
        # warnings
        major, minor = torch.cuda.get_device_capability()
        arch_string = f"{major}.{minor}"

        # Set environment variable to suppress warning
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_string

        # Default compiler flags
        if extra_cflags is None:
            extra_cflags = ["-O3"]

        if extra_cuda_cflags is None:
            arch_flags = [f"-gencode=arch=compute_{major}{minor}," f"code=sm_{major}{minor}"]

            extra_cuda_cflags = [
                "-O3",
                "-use_fast_math",
                "-lineinfo",
            ] + arch_flags

        logger.info(f"Compiling CUDA kernel: {name}")
        logger.info(f"Compiling for GPU architecture: sm_{major}{minor}")

        module = load_inline(
            name=name,
            cpp_sources=[source_code],
            cuda_sources=[],
            functions=[],  # Will be defined in source_code via PYBIND11_MODULE
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )

        self._compiled_modules[name] = module
        logger.info(f"CUDA kernel {name} compiled successfully")

        return module

    def compile_from_file(
        self,
        cuda_file: str,
        name: str,
        extra_cflags=None,
        extra_cuda_cflags=None,
    ) -> Any:
        """Compile CUDA kernel from file.

        Args:
            cuda_file: Path to CUDA source file (relative to cc/ directory)
            name: Name for the compiled module
            extra_cflags: Additional C++ compiler flags
            extra_cuda_cflags: Additional CUDA compiler flags

        Returns:
            Compiled CUDA module
        """
        if name in self._compiled_modules:
            return self._compiled_modules[name]

        cuda_path = self._cc_dir / cuda_file
        if not cuda_path.exists():
            raise FileNotFoundError(f"CUDA source file not found: {cuda_path}")

        try:
            import torch
            from torch.utils.cpp_extension import load
        except ImportError:
            raise ImportError("PyTorch is required for CUDA kernel compilation")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Detect GPU architecture and set TORCH_CUDA_ARCH_LIST to avoid
        # warnings
        major, minor = torch.cuda.get_device_capability()
        arch_string = f"{major}.{minor}"

        # Set environment variable to suppress warning
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_string

        # Default compiler flags
        if extra_cflags is None:
            extra_cflags = ["-O3"]

        if extra_cuda_cflags is None:
            arch_flags = [f"-gencode=arch=compute_{major}{minor}," f"code=sm_{major}{minor}"]

            extra_cuda_cflags = [
                "-O3",
                "-use_fast_math",
                "-lineinfo",
            ] + arch_flags

        logger.info(f"Compiling CUDA kernel: {name}")
        logger.info(f"Compiling for GPU architecture: sm_{major}{minor}")

        # Use load instead of load_inline for file-based compilation
        module = load(
            name=name,
            sources=[str(cuda_path)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )

        self._compiled_modules[name] = module
        logger.info(f"CUDA kernel {name} compiled successfully")

        return module

    def get_compiled_module(self, name: str) -> Optional[Any]:
        """Get previously compiled module.

        Args:
            name: Name of the compiled module

        Returns:
            Compiled module if exists, None otherwise
        """
        return self._compiled_modules.get(name)

    def clear_cache(self):
        """Clear compiled module cache."""
        self._compiled_modules.clear()


# Global compiler instance
_compiler = CUDAKernelCompiler()


def get_cuda_compiler() -> CUDAKernelCompiler:
    """Get the global CUDA kernel compiler instance."""
    return _compiler


def compile_softmax_kernel() -> Any:
    """Compile SoftMax CUDA kernel.

    Returns:
        Compiled CUDA module with forward/backward functions
    """
    return _compiler.compile_from_file("softmax_cuda.cu", "softmax_cuda")
