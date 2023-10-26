// Copyright 2023 The TFPlus Authors. All rights reserved.
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

#ifndef TFPLUS_KV_VARIABLE_KERNELS_UTILITY_H_
#define TFPLUS_KV_VARIABLE_KERNELS_UTILITY_H_
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "farmhash.h"  // NOLINT(build/include_subdir)
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tfplus {
using ::tensorflow::uint16;
using ::tensorflow::Status;
static const std::vector<std::string> REPARTITION_TENSORS_SUFFIX = {
    "-keys",      "-values",      "-blacklist",
    "-freq_keys", "-freq_values", "-init_table"};

// Get current unix time, by default divided by 3600*24 as days.
uint16 GetCurrentUnixTimeByDivisor(uint64_t divisor = 3600 * 24);

uint16_t GetUint16FromUint32(const uint32_t& source, bool isLow16Bits);

uint32_t MakeUint32FromUint16(uint16_t high, uint16_t low);

// Saturate arithmetic operation.
inline uint16_t SaturateMaxFrequency(const int32_t freq) {
  return std::min<int32_t>(freq, std::numeric_limits<uint16_t>::max());
}

bool HasPublicKeys(const int64_t old_num_shards, const int64_t new_num_shards,
                   const int64_t old_num_index, const int64_t new_num_index);

// Saturate arithmetic addition operation.
inline uint16_t SaturateAddFrequency(const uint16_t val, const uint16_t delta) {
  uint16_t new_val = val + delta;
  if (new_val < val) /* Can only happen due to overflow */
    new_val = -1;
  return new_val;
}

bool IsLegacyScalar(const ::tensorflow::TensorShape& shape);

template <typename T>
inline T StringToValue(const char* value, T default_if_failed) {
  std::istringstream ss(value);
  T res;
  ss >> res;
  if (ss.fail()) {
    return default_if_failed;
  }
  return res;
}

template <typename T>
inline T GetEnvVar(const char* key, T val) {
  char* name = getenv(key);
  if (name != nullptr && strlen(name) > 0) {
    return StringToValue<T>(name, val);
  } else {
    return val;
  }
}

template <typename T>
struct google_floor_mod {
  const inline T operator()(const T& x, const int& y) const {
    T trunc_mod = x % y;
    return (x < T(0)) == (y < T(0)) ? trunc_mod : (trunc_mod + y) % y;
  }
};

template <typename TL>
inline int ModKeyImpl(const TL& key, const int& num_shards) {
  google_floor_mod<TL> floor_mod;
  return floor_mod(key, num_shards);
}

template <>
inline int ModKeyImpl(const std::string& key, const int& num_shards) {
  return ::util::Fingerprint64<std::string>(key) % num_shards;
}

int64_t Gcd(int64_t m, int64_t n);

std::string GenerateSnapshotPath(const std::string& prefix);

std::string RemoveCheckpointPathTempSuffix(const std::string& path);

std::string ConcatStringList(const std::vector<std::string>& string_list);

bool CanMakeSymlink(const std::vector<std::string>& pathnames,
                    const std::string& dest_dir);


Status Unlink(const std::string& path);

void* AllocateRaw(size_t size);

template <typename T>
void DeallocateRaw(T* ptr) {
  ::tensorflow::cpu_allocator()->DeallocateRaw(reinterpret_cast<void*>(ptr));
}

}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_KERNELS_UTILITY_H_
