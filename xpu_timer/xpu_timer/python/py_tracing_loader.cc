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

#include "xpu_timer/python/py_tracing_loader.h"

#include <dlfcn.h>

#include <cstring>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/macro.h"

namespace xpu_timer {
namespace py_tracing_manager {

PyTracingLibrary::PyTracingLibrary(const std::string& library_path)
    : LibraryLoader(library_path),
      register_tracing_(nullptr),
      get_tracing_data_(nullptr),
      get_partial_tracing_data_(nullptr),
      return_tracing_data_(nullptr),
      switch_tracing_(nullptr) {
  const std::string err =
      "libpy_tracing.so, skip recording python gc in timeline ";
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "xpu_timer_register_tracing",
                                register_tracing_, XpuTimerRegisterTracingFunc,
                                err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(
      handle_, "xpu_timer_get_full_py_tracing_data_array", get_tracing_data_,
      GetFullTracingDataArrayFunc, err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(
      handle_, "xpu_timer_return_py_tracing_data_array", return_tracing_data_,
      ReturnTracingDataArrayFunc, err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(
      handle_, "xpu_timer_get_partial_py_tracing_data_array",
      get_partial_tracing_data_, GetPartialTracingDataArrayFunc, err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "xpu_timer_switch_py_tracing",
                                switch_tracing_, SwitchTracingFunc, err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "xpu_timer_get_py_tracing_count",
                                get_tracing_count_, GetTracingCountFunc, err);
  XLOG(INFO) << "Load libgc_callback.so ok";
  can_use_ = true;
}

std::vector<std::string> PyTracingLibrary::Register(
    const std::vector<std::string>& names) {
  if (!can_use_) return {};
  std::vector<std::string> result;
  // this take own ship of all errors
  char** errors = (char**)malloc(names.size() * sizeof(char*));
  std::memset(errors, 0, names.size() * sizeof(char*));

  std::vector<const char*> c_str_array;
  for (const auto& str : names) {
    c_str_array.push_back(str.c_str());
  }
  register_tracing_(c_str_array.data(), c_str_array.size(), errors);
  for (size_t i = 0; i < names.size(); i++) {
    if (errors[i]) {
      result.push_back(std::string(errors[i]));
      free(errors[i]);
    }
  }

  free(errors);
  return result;
}

int64_t PyTracingLibrary::GetTracingCount(int name) {
  if (can_use_) return get_tracing_count_(name);
  return -1;
}

XpuTimerPyTracingDataArray* PyTracingLibrary::GetFullTracingData(int name) {
  if (can_use_) {
    return get_tracing_data_(name);
  }
  return nullptr;
}

XpuTimerPyTracingDataArray* PyTracingLibrary::GetPartialTracingData(int name) {
  if (can_use_) {
    return get_partial_tracing_data_(name);
  }
  return nullptr;
}

void PyTracingLibrary::ReturnTracingData(XpuTimerPyTracingDataArray* data,
                                         int type, int name) {
  if (can_use_ && data) return_tracing_data_(data, type, name);
}

void PyTracingLibrary::SwitchTracing(int flag) {
  if (can_use_) switch_tracing_(flag);
}

}  // namespace py_tracing_manager
}  // namespace xpu_timer
