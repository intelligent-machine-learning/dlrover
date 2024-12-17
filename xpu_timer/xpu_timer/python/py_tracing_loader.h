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
#include <vector>

#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/python/py_tracing_data.h"

namespace xpu_timer {
namespace py_tracing_manager {

class PyTracingLibrary : public LibraryLoader {
 public:
  PyTracingLibrary(const std::string&);
  using XpuTimerRegisterTracingFunc = void (*)(const char**, int, char**);
  using GetFullTracingDataArrayFunc = XpuTimerPyTracingDataArray* (*)(int);
  using GetPartialTracingDataArrayFunc = XpuTimerPyTracingDataArray* (*)(int);
  using ReturnTracingDataArrayFunc = void (*)(XpuTimerPyTracingDataArray*, int,
                                              int);
  using SwitchTracingFunc = void (*)(int);
  using GetTracingCountFunc = int64_t (*)(int);

  std::vector<std::string> Register(const std::vector<std::string>& names);
  XpuTimerPyTracingDataArray* GetFullTracingData(int);
  XpuTimerPyTracingDataArray* GetPartialTracingData(int);
  void ReturnTracingData(XpuTimerPyTracingDataArray* data, int type, int name);
  void SwitchTracing(int flag);
  int64_t GetTracingCount(int name);

 private:
  XpuTimerRegisterTracingFunc register_tracing_;
  GetFullTracingDataArrayFunc get_tracing_data_;
  GetPartialTracingDataArrayFunc get_partial_tracing_data_;
  ReturnTracingDataArrayFunc return_tracing_data_;
  SwitchTracingFunc switch_tracing_;
  GetTracingCountFunc get_tracing_count_;
};

}  // namespace py_tracing_manager
}  // namespace xpu_timer
