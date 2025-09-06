#pragma once
#include <unordered_map>

#include "xpu_timer/common/platform.h"

namespace xpu_timer {
namespace hpu {

class HpuDataTypeUtils {
 public:
  static const std::string UNKNOWN_ACL_DTYPE;

  static const std::string& getAclDtype(aclDataType dtype);
  static const std::string& getHcclDataType(const HcclDataType& dtype);
  static uint64_t getDtypeSizeInBytes(const std::string& dtype);
  static double getGpuHardwareFlops(const std::string& dtype);
  static void setGpu(const std::string& gpu);

 private:
  static const std::unordered_map<aclDataType, std::string>
      aclDataTypeToStringMap;
  static const std::unordered_map<HcclDataType, std::string>
      hcclDataTypeToStringMap;
  static const std::unordered_map<std::string, uint64_t> dtypeSizeInBytes;
  static const std::unordered_map<std::string,
                                  std::unordered_map<std::string, double>>
      gpuHardwareFlops;
  static std::string gpu_;
};
}  // namespace hpu
}  // namespace xpu_timer
