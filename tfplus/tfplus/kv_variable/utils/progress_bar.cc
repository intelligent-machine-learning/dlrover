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

#include "tfplus/kv_variable/utils/progress_bar.h"

#include <map>
#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_set.h"
#include "tfplus/kv_variable/kernels/mutex.h"

namespace tfplus {
using ::tensorflow::mutex;
using tensorflow::mutex_lock;
ProgressBar::ProgressBar(std::string process_name, size_t task_num,
                         int every_report_percent, bool report_every_task,
                         int min_report_secs, int max_report_secs)
    : process_name_(process_name),
      task_num_(task_num),
      every_report_percent(every_report_percent),
      report_every_task_(report_every_task),
      min_report_secs_(min_report_secs),
      max_report_secs_(max_report_secs),
      task_finish_num_(0) {
  start_time_ =::tensorflow::Env::Default()->NowMicros();
}

void ProgressBar::FinishTask(std::string task_name) {
  mutex_lock l(mu_);
  task_finish_num_++;
  if (report_every_task_) {
    VLOG(0) << "Process " << process_name_ << " " << task_name << " finish "
            << MakeProgressBar();
  }
  if (task_finish_num_ == task_num_) {
    VLOG(0) << "Process " << process_name_ << " finish, cost time "
            <<::tensorflow::Env::Default()->NowMicros() - start_time_;
  }
}

std::string ProgressBar::MakeProgressBar(bool make_by_task) {
  int character_num = 32;
  int complete_character_num =
      1.0 * task_finish_num_ / task_num_ * character_num;
  int incomplete_character_num = character_num - complete_character_num;
  std::string res = "|";
  while (complete_character_num--) {
    res += complete_character_;
  }
  while (incomplete_character_num--) {
    res += incomplete_character_;
  }
  res +=
      "| " + std::to_string(task_finish_num_) + "/" + std::to_string(task_num_);
  return res;
}
}  // namespace tfplus

