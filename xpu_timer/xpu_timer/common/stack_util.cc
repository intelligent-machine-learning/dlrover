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

#include "xpu_timer/common/stack_util.h"

#include <butil/logging.h>
#include <dlfcn.h>

#include <fstream>
#include <memory>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/protos/hosting_service.pb.h"

namespace xpu_timer {
namespace stack_util {

using namespace util;
using namespace server;

PyStackInProcess::PyStackInProcess(const std::string& library_path)
    : LibraryLoader(library_path), get_py_stack_fn_(nullptr), dump_count_(0) {
  total_dump_count_ =
      util::EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_DUMP_STACK_COUNT");

  // libpy_xpu_timer_callstack.so is in same dir with libevent_hook.so, and set
  // rpath,ORIGIN
  const std::string err =
      "libpy_xpu_timer_callstack.so, skip dump python stack in timeline";
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "gather_python_callstack",
                                get_py_stack_fn_, getPyStackFn, err);
  XLOG(INFO) << "Load fn `gather_python_callstack` OK";
  can_use_ = true;
  if (total_dump_count_ == 0) {
    can_use_ = false;
    XLOG(INFO) << "Dump stack count is 0, disable dump";
  }
};

PyStack PyStackInProcess::getPyStack() {
  if (!can_use_) return {};
  return get_py_stack_fn_();
}

bool PyStackInProcess::isFull() {
  if (!can_use_) return true;
  return dump_count_ > total_dump_count_;
}

void PyStackInProcess::insertPyStack(const std::string& kernel_name,
                                     const PyStack& stack) {
  if (!can_use_) return;
  // not thread safe, pytorch is single thread launch kernel, maybe add lock in
  // future.
  // if kernel_name is same, we only peak first callstack, may be add hash of
  // vector<stack> to hold all stacks.
  if (stack.empty()) return;
  auto it = stack_maps_.find(kernel_name);
  if (it == stack_maps_.end()) {
    stack_maps_.emplace(std::piecewise_construct,
                        std::forward_as_tuple(kernel_name),
                        std::forward_as_tuple(std::move(stack)));
  }
  dump_count_++;
  if (dump_count_ == total_dump_count_) XLOG(INFO) << "Kernel trace dump ready";
}

void PyStackInProcess::dumpPyStack(const std::string& path, int rank) {
  if (!can_use_) {
    XLOG(INFO) << "Skip Dump timeline stack to " << path;
    return;
  }
  PythonStackInTimeline timeline;
  timeline.set_rank(rank);
  auto& frames_map = *timeline.mutable_named_frames();
  for (const auto& kernel_stack : stack_maps_) {
    for (int i = kernel_stack.second.size() - 1; i >= 0; i--) {
      auto& stack = kernel_stack.second[i];
      PySpyFrame* frame = frames_map[kernel_stack.first].add_frames();
      frame->set_func_name(stack.function_name);
      frame->set_file_name(stack.filename + ":" + std::to_string(stack.line));
    }
  }
  std::string binary_message;
  timeline.SerializeToString(&binary_message);
  std::string file_path =
      path + "/" +
      util::getUniqueFileNameByCluster(".tracing_kernel_callstack");
  std::ofstream file(file_path);
  file << binary_message;
  XLOG(INFO) << "Dump timeline stack to " << file_path;
}

}  // namespace stack_util
}  // namespace xpu_timer
