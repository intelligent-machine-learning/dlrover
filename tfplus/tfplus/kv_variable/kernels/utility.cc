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

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <utility>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <regex>  // NOLINT(build/c++11)
#include <cmath>
#include "tensorflow/core/platform/logging.h"
#include "tfplus/kv_variable/kernels/utility.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/io/path.h"

namespace tfplus {
using ::tensorflow::uint16;
constexpr int AntKCopyFileBufferSize = 8 << 20;  // 8MB buffer for dfs

bool IsLegacyScalar(const ::tensorflow::TensorShape& shape) {
  return shape.dims() == 0 || (shape.dims() == 1 &&
                                shape.dim_size(0) == 1);
}

// Get current unix time, by default divided by 3600*24 as days.
uint16 GetCurrentUnixTimeByDivisor(uint64_t divisor) {
  return static_cast<uint16>(std::time(nullptr) / divisor);
}

uint16_t GetUint16FromUint32(const uint32_t& source, bool isLow16Bits) {
  return isLow16Bits ? static_cast<uint16_t>(source & 0xFFFF)
                     : static_cast<uint16_t>(source >> 16);
}

uint32_t MakeUint32FromUint16(uint16_t high, uint16_t low) {
  return (((uint32_t)high << 16) | (uint32_t)low);
}

std::string ConcatStringList(const std::vector<std::string>& string_list) {
  std::string result = "[";
  for (auto&& s : string_list) {
    result += s + ", ";
  }
  result += "]";
  return result;
}

int GetPathLevel(const std::string& path) {
  return std::count(path.begin(), path.end(), '/');
}

int64_t Gcd(int64_t m, int64_t n) {
  return std::__gcd(m, n);
}

bool HasPublicKeys(const int64_t old_num_shards, const int64_t new_num_shards,
                   const int64_t old_num_index, const int64_t new_num_index) {
  auto g = Gcd(old_num_shards, new_num_shards);
  return (old_num_index - new_num_index) % g == 0;
}

std::string GenerateSnapshotPath(const std::string& prefix) {
  return prefix + ".snapshot";
}

std::string RemoveCheckpointPathTempSuffix(const std::string& path) {
  // example: _temp_d1b6a51df8a84b92a12ffa7bf271437a/part-00000-of-00020
  std::regex suffix_regex("_temp_[\\da-f]{32}/part-[\\d]{5}-of-[\\d]{5}$");
  std::cmatch m;
  if (!std::regex_search(path.c_str(), m, suffix_regex)) {
    return path;
  } else {
    return m.prefix();;
  }
}

bool CanMakeSymlink(const std::vector<std::string>& pathnames,
                    const std::string& dest_dir) {
  if (dest_dir.rfind("/", 0) != 0) {
    LOG(WARNING) << "can't make symbolic link because directory " << dest_dir
                 << " is not a local directory";
    return false;
  }
  std::regex schema("(dfs)|(pangu)|(oss)");
  for (auto& pathname : pathnames) {
    if (pathname.rfind("/", 0) != 0) {
      LOG(WARNING) << "can't make symbolic link because " << pathname
                   << " is not a local file";
      return false;
    }
    std::cmatch m;
    if (std::regex_search(pathname.c_str(), m, schema)) {
      LOG(WARNING) << "can't make symbolic link because " << pathname
                   << " is not a local file";
      return false;
    }
    struct stat sb;
    if (!(stat(pathname.c_str(), &sb) == 0 &&
          (S_ISREG(sb.st_mode) || S_ISDIR(sb.st_mode)))) {
      LOG(WARNING) << "can't make symbolic link because " << pathname
                   << " is not a local file";
      return false;
    }
  }
  return true;
}

Status Unlink(const std::string& path) {
  // if (unlink(path.c_str()) != 0) {
  //   return ::tensorflow::errors::Internal("Failed to unlink file: ", path);
  // }
  return ::tensorflow::OkStatus();
}

void* AllocateRaw(size_t size) {
  return ::tensorflow::cpu_allocator()->AllocateRaw(
      ::tensorflow::Allocator::kAllocatorAlignment, size);
}

}  // namespace tfplus
