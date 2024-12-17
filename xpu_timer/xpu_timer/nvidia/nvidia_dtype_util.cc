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

#include "xpu_timer/nvidia/nvidia_dtype_util.h"

namespace xpu_timer {
namespace nvidia {

const std::string CudaDataTypeUtils::UNKNOWN_CUDA_DTYPE = "UNKNOWN";
std::string CudaDataTypeUtils::gpu_ = "";

const std::unordered_map<cudaDataType_t, std::string>
    CudaDataTypeUtils::cudaDataTypeToStringMap = {
        {CUDA_R_16F, "fp16"},         {CUDA_C_16F, "complex_fp16"},
        {CUDA_R_16BF, "bf16"},        {CUDA_C_16BF, "complex_bf16"},
        {CUDA_R_32F, "fp32"},         {CUDA_C_32F, "complex_fp32"},
        {CUDA_R_64F, "fp64"},         {CUDA_C_64F, "complex_fp64"},
        {CUDA_R_8I, "int8"},          {CUDA_C_8I, "complex_int8"},
        {CUDA_R_8U, "uint8"},         {CUDA_C_8U, "complex_uint8"},
        {CUDA_R_32I, "int32"},        {CUDA_C_32I, "complex_int32"},
#if defined(XPU_NVIDIA) && defined(CUDA_FP8)
        {CUDA_R_8F_E4M3, "fp8_e4m3"}, {CUDA_R_8F_E5M2, "fp8_e5m2"},
#endif
};

const std::unordered_map<ncclDataType_t, std::string>
    CudaDataTypeUtils::ncclDataTypeToStringMap = {
        {ncclInt8, "int8"},
        {ncclChar, "int8"},
        {ncclUint8, "uint8"},
        {ncclInt32, "int32"},
        {ncclInt, "int32"},
        {ncclUint32, "uint32"},
        {ncclInt64, "int64"},
        {ncclUint64, "uint64"},
        {ncclFloat16, "fp16"},
        {ncclHalf, "fp16"},
        {ncclFloat32, "fp32"},
        {ncclFloat, "fp32"},
        {ncclFloat64, "fp64"},
        {ncclDouble, "fp64"},
#if defined(__CUDA_BF16_TYPES_EXIST__)
        {ncclBfloat16, "bf16"},
#endif
        {ncclNumTypes, UNKNOWN_CUDA_DTYPE},
};

const std::unordered_map<std::string, uint64_t>
    CudaDataTypeUtils::dtypeSizeInBytes = {
        {"fp16", 2},
        {"bf16", 2},
        {"fp64", 8},
        {"fp32", 4},
        {"int8", 1},
        {"int32", 4},
        {"int64", 8},
        {"uint8", 1},
        {"uint32", 4},
        {"uint64", 8},
        {UNKNOWN_CUDA_DTYPE, 0},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>>
    CudaDataTypeUtils::gpuHardwareFlops = {
        {"P100",
         {
             {"fp16", 19.05},
             {"bf16", 19.05},
             {"fp32", 9.526},
             {"fp64", 4.763},
         }},
        {"A100",
         {
             {"fp16", 311.84},
             {"bf16", 311.84},
             {"fp32", 19.49},
             {"fp64", 9.746},
         }},
        {"A800",
         {
             {"fp16", 311.84},
             {"bf16", 311.84},
             {"fp32", 19.49},
             {"fp64", 9.746},
         }},
        {"H100",
         {
             {"fp16", 989.},
             {"bf16", 989.},
             {"fp32", 67.},
             {"fp64", 67.},
             {"fp8_e4m3", 1979.},
             {"fp8_e5m2", 1979.},
         }},
        {"H800",
         {
             {"fp16", 989.},
             {"bf16", 989.},
             {"fp32", 67.},
             {"fp64", 67.},
             {"fp8_e4m3", 1979.},
             {"fp8_e5m2", 1979.},
         }},
        {"V100",
         {
             {"fp16", 125.},
             {"bf16", 125.},
             {"fp32", 15.7},
             {"fp64", 7.8},
         }},
};

// Implementations of static methods
const std::string& CudaDataTypeUtils::getCudaDtype(cudaDataType_t dtype) {
  auto it = cudaDataTypeToStringMap.find(dtype);
  return it == cudaDataTypeToStringMap.end() ? UNKNOWN_CUDA_DTYPE : it->second;
}

const std::string& CudaDataTypeUtils::getNcclDataType(
    const ncclDataType_t& dtype) {
  auto it = ncclDataTypeToStringMap.find(dtype);
  return it == ncclDataTypeToStringMap.end() ? UNKNOWN_CUDA_DTYPE : it->second;
}

uint64_t CudaDataTypeUtils::getDtypeSizeInBytes(const std::string& dtype) {
  auto it = dtypeSizeInBytes.find(dtype);
  return it == dtypeSizeInBytes.end() ? 0 : it->second;
}

void CudaDataTypeUtils::setGpu(const std::string& gpu) { gpu_ = gpu; }

double CudaDataTypeUtils::getGpuHardwareFlops(const std::string& dtype) {
  static const std::unordered_map<std::string, double>* gpu_ptr = nullptr;
  if (!gpu_ptr) {
    auto it = gpuHardwareFlops.find(gpu_);
    if (it != gpuHardwareFlops.end()) {
      gpu_ptr = &it->second;
    } else {
      gpu_ptr = &gpuHardwareFlops.at("A100");
    }
  }

  auto it = gpu_ptr->find(dtype);
  if (it != gpu_ptr->end()) {
    return it->second;
  }
  // defaults to hafl on A100
  return 312.;
}

}  // namespace nvidia
}  // namespace xpu_timer
