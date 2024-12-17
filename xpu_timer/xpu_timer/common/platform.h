// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <string>

namespace xpu_timer {
namespace platform {
std::string getDeviceName();
}
}  // namespace xpu_timer

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
