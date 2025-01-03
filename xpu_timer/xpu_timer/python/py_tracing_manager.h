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
#include <iostream>
#include <string>
#include <unordered_map>

#include "xpu_timer/common/util.h"
#include "xpu_timer/python/py_tracing.h"
#include "xpu_timer/python/py_tracing_data.h"

namespace xpu_timer {
namespace py_tracing_manager {

class PyTracingManager {
 public:
  PyTracingManager(const PyTracingManager&) = delete;
  PyTracingManager& operator=(const PyTracingManager&) = delete;
  static void initSingleton();
  static PyTracingManager& getInstance();

  XpuTimerPyTracingDataArray* getEmptyPyTracingDataArray(int name);
  void returnPyTracingDataArray(XpuTimerPyTracingDataArray*, int, int name);
  XpuTimerPyTracingDataArray* getPyTracingDataArray(int name);
  XpuTimerPyTracingDataArray* getCurPyTracingDataArray(int name);

 private:
  PyTracingManager() = default;
  inline static PyTracingManager* instance_ = nullptr;
  inline static std::once_flag init_flag_;
  struct Pool {
    util::TimerPool<XpuTimerPyTracingDataArray> empty_pool;
    util::TimerPool<XpuTimerPyTracingDataArray> ready_pool;
  };
  std::unordered_map<int, Pool> pool_;
};
}  // namespace py_tracing_manager
}  // namespace xpu_timer

#ifdef __cplusplus
extern "C" {
#endif
XpuTimerPyTracingDataArray* xpu_timer_get_empty_py_tracing_data_array(
    int name) {
  return xpu_timer::py_tracing_manager::PyTracingManager::getInstance()
      .getEmptyPyTracingDataArray(name);
}

XpuTimerPyTracingDataArray* xpu_timer_get_full_py_tracing_data_array(int name) {
  return xpu_timer::py_tracing_manager::PyTracingManager::getInstance()
      .getPyTracingDataArray(name);
}

void xpu_timer_return_py_tracing_data_array(XpuTimerPyTracingDataArray* array,
                                            int type, int name) {
  xpu_timer::py_tracing_manager::PyTracingManager::getInstance()
      .returnPyTracingDataArray(array, type, name);
}

#ifdef __cplusplus
}
#endif
