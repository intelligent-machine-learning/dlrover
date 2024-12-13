#pragma once
#include <string>

namespace xpu_timer {
namespace platform {
std::string getDeviceName();
}
} // namespace xpu_timer

#if defined(XPU_NVIDIA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#if defined(CUDART_VERSION)

#if CUDART_VERSION >= 11000
#define CUDA_VERSION_11
#endif

#if CUDART_VERSION >= 11080
#define CUDA_VERSION_11_8
#define CUDA_LAUNCH_EXC
#define CUDA_FP8
#endif

#if CUDART_VERSION >= 12000
#define CUDA_VERSION_12
#endif

#else
#error "CUDART_VERSION is not defined; ensure CUDA runtime is included."
#endif

#endif
