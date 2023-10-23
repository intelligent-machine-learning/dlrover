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

#ifndef TFPLUS_KV_VARIABLE_UTILS_PROGRESS_BAR_H_
#define TFPLUS_KV_VARIABLE_UTILS_PROGRESS_BAR_H_

#include <map>
#include <string>
#include <unordered_map>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tfplus/kv_variable/kernels/mutex.h"
namespace tfplus {
using ::tensorflow::mutex;
class ProgressBar {
 public:
  struct TaskInfo {
    explicit TaskInfo(std::string task_name) : task_name_(task_name) {
      start_time_ = tensorflow::Env::Default()->NowMicros();
    }
    std::string task_name_;
    size_t start_time_;
  };

  ProgressBar(std::string process_name, size_t task_num,
              int every_report_percent = 10, bool report_every_task = true,
              int min_report_secs = 0, int max_report_secs = 0);

  void FinishTask(std::string task_name);

  std::string MakeProgressBar(bool make_by_task = true);

  mutable mutex mu_;
  size_t task_num_;
  size_t start_time_;
  size_t task_finish_num_;
  int every_report_percent;
  int min_report_secs_;
  int max_report_secs_;
  bool report_every_task_;
  std::string process_name_;
  std::string complete_character_ = "█";
  std::string incomplete_character_ = " ";
};

}  // namespace tfplus

#endif  // TFPLUS_KV_VARIABLE_UTILS_PROGRESS_BAR_H_
