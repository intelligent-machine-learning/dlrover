#pragma once

#include <string>
#include <unordered_map>

#include "xpu_timer/common/platform.h"
namespace xpu_timer {
namespace nvidia {

class CudaDataTypeUtils {
public:
  static const std::string UNKNOWN_CUDA_DTYPE;

  static const std::string &getCudaDtype(cudaDataType_t dtype);
  static const std::string &getNcclDataType(const ncclDataType_t &dtype);
  static uint64_t getDtypeSizeInBytes(const std::string &dtype);
  static double getGpuHardwareFlops(const std::string &dtype);
  static void setGpu(const std::string &gpu);

private:
  static const std::unordered_map<cudaDataType_t, std::string>
      cudaDataTypeToStringMap;
  static const std::unordered_map<ncclDataType_t, std::string>
      ncclDataTypeToStringMap;
  static const std::unordered_map<std::string, uint64_t> dtypeSizeInBytes;
  static const std::unordered_map<std::string,
                                  std::unordered_map<std::string, double>>
      gpuHardwareFlops;
  static std::string gpu_;
};

} // namespace nvidia
} // namespace xpu_timer
