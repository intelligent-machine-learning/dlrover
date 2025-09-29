
#include "xpu_timer/hpu/hpu_dtype_util.h"

namespace xpu_timer {
namespace hpu {
const std::string HpuDataTypeUtils::UNKNOWN_ACL_DTYPE = "UNKNOWN";
std::string HpuDataTypeUtils::gpu_ = "";

const std::unordered_map<aclDataType, std::string>
    HpuDataTypeUtils::aclDataTypeToStringMap = {
        {ACL_FLOAT, "float32"},       {ACL_FLOAT16, "float16"},
        {ACL_INT8, "int8"},           {ACL_UINT8, "uint8"},
        {ACL_INT16, "int16"},         {ACL_UINT16, "uint16"},
        {ACL_UINT32, "uint32"},       {ACL_INT64, "int64"},
        {ACL_UINT64, "uint64"},       {ACL_DOUBLE, "double"},
        {ACL_BOOL, "bool"},           {ACL_STRING, "string"},
        {ACL_COMPLEX64, "complex64"}, {ACL_COMPLEX128, "complex128"},
        {ACL_BF16, "bfloat16"},       {ACL_INT4, "int4"},
        {ACL_UINT1, "uint1"},         {ACL_COMPLEX32, "complex32"},
};

const std::unordered_map<HcclDataType, std::string>
    HpuDataTypeUtils::hcclDataTypeToStringMap = {
        {HCCL_DATA_TYPE_INT8, "int8"},      /* int8 */
        {HCCL_DATA_TYPE_INT16, "int16"},    /* int16 */
        {HCCL_DATA_TYPE_INT32, "int32"},    /* int32 */
        {HCCL_DATA_TYPE_FP16, "float16"},   /* fp16 */
        {HCCL_DATA_TYPE_FP32, "float32"},   /* fp32 */
        {HCCL_DATA_TYPE_INT64, "int64"},    /* int64 */
        {HCCL_DATA_TYPE_UINT64, "uint64"},  /* uint64 */
        {HCCL_DATA_TYPE_UINT8, "uint8"},    /* uint8 */
        {HCCL_DATA_TYPE_UINT16, "uint16"},  /* uint16 */
        {HCCL_DATA_TYPE_UINT32, "uint32"},  /* uint32 */
        {HCCL_DATA_TYPE_FP64, "float64"},   /* fp64 */
        {HCCL_DATA_TYPE_BFP16, "bfloat16"}, /* bfp16 */
        {HCCL_DATA_TYPE_INT128, "int128"},  /* int128 */
};

const std::unordered_map<std::string, uint64_t>
    HpuDataTypeUtils::dtypeSizeInBytes = {
        {"int8", 1},    {"int16", 2},           {"int32", 4},   {"float16", 2},
        {"float32", 4}, {"int64", 8},           {"uint64", 8},  {"uint8", 1},
        {"uint16", 2},  {"uint32", 4},          {"float64", 8}, {"bfloat16", 2},
        {"int128", 16}, {UNKNOWN_ACL_DTYPE, 0},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>>
    HpuDataTypeUtils::gpuHardwareFlops = {
        {"910B",
         {
             {"fp16", 376.350},
             {"bf16", 364.928},
             {"fp32", 99.559},
         }},
};
const std::string& HpuDataTypeUtils::getAclDtype(aclDataType dtype) {
  auto it = aclDataTypeToStringMap.find(dtype);
  return it == aclDataTypeToStringMap.end() ? UNKNOWN_ACL_DTYPE : it->second;
}

uint64_t HpuDataTypeUtils::getDtypeSizeInBytes(const std::string& dtype) {
  auto it = dtypeSizeInBytes.find(dtype);
  return it == dtypeSizeInBytes.end() ? 0 : it->second;
}

void HpuDataTypeUtils::setGpu(const std::string& gpu) { gpu_ = gpu; }

double HpuDataTypeUtils::getGpuHardwareFlops(const std::string& dtype) {
  static const std::unordered_map<std::string, double>* gpu_ptr = nullptr;
  if (!gpu_ptr) {
    auto it = gpuHardwareFlops.find(gpu_);
    if (it != gpuHardwareFlops.end()) {
      gpu_ptr = &it->second;
    } else {
      gpu_ptr = &gpuHardwareFlops.at("910B");
    }
  }

  auto it = gpu_ptr->find(dtype);
  if (it != gpu_ptr->end()) {
    return it->second;
  }
  // 910B half
  return 376.250;
}

const std::string& HpuDataTypeUtils::getHcclDataType(
    const HcclDataType& dtype) {
  auto it = hcclDataTypeToStringMap.find(dtype);
  return it == hcclDataTypeToStringMap.end() ? UNKNOWN_ACL_DTYPE : it->second;
}

}  // namespace hpu
}  // namespace xpu_timer
