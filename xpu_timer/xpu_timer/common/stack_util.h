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
#include <unordered_map>
#include <vector>

#include "xpu_timer/common/xpu_timer.h"

namespace xpu_timer {
namespace stack_util {

class PyStackInProcess : public LibraryLoader {
 private:
  using getPyStackFn = PyStack (*)();
  getPyStackFn get_py_stack_fn_;
  uint64_t dump_count_;
  std::unordered_map<std::string, PyStack> stack_maps_;
  void LoadFn();
  uint64_t total_dump_count_;

 public:
  PyStackInProcess(const std::string& library_path);
  void insertPyStack(const std::string& kernel_name, const PyStack& stack);
  void dumpPyStack(const std::string& path, int rank);
  bool isFull();
  PyStack getPyStack();
};

}  // namespace stack_util
}  // namespace xpu_timer
