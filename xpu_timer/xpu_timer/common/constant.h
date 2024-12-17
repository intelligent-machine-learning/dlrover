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
#include <algorithm>

namespace xpu_timer {
namespace constant {

static constexpr bool SKIP_TP = true;

struct Metrics {
 public:
  static constexpr int AVG_LATENCY = 0;
  static constexpr int MAX_LATENCY = 1;
  static constexpr int P99_LATENCY = 2;
  static constexpr int MIN_LATENCY = 3;
  static constexpr int PERFORMANCE = 4;

  static constexpr int HANG = 0;
  static constexpr int START_DUMP = 1;
  static constexpr int END_DUMP = 2;
  static constexpr int POOL_QUEUE_SIZE = 3;
  static constexpr int WORK_QUEUE_SIZE = 4;

  static constexpr int COUNTER = 0;

  struct CollMetrics {
    static constexpr uint32_t KERNEL_TYPE = 1;
    static constexpr std::string_view BUCKET_NAME = "COLL_P";
    static constexpr std::string_view TYPE = "coll";
    static constexpr int CAP = 5;
    static constexpr std::string_view GAUGE_PREFIX = "XPU_TIMER_COLL_KERNEL_";
    static constexpr std::string_view METRICS_NAME[CAP] = {
        "AVG_LATENCY", "MAX_LATENCY", "P99_LATENCY", "MIN_LATENCY",
        "BANDWIDTH"};
  };
  struct MatmulMetrics {
    static constexpr uint32_t KERNEL_TYPE = 0;
    static constexpr std::string_view BUCKET_NAME = "COMPUTE_P";
    static constexpr std::string_view TYPE = "mm";
    static constexpr int CAP = 5;
    static constexpr std::string_view GAUGE_PREFIX = "XPU_TIMER_MM_KERNEL_";
    static constexpr std::string_view METRICS_NAME[CAP] = {
        "AVG_LATENCY", "MAX_LATENCY", "P99_LATENCY", "MIN_LATENCY", "FLOPS"};
  };
  struct CommonMetrics {
    static constexpr std::string_view TYPE = "common";
    static constexpr std::string_view GAUGE_PREFIX = "XPU_TIMER_COMMON_";
    static constexpr int CAP = 7;
    static constexpr std::string_view METRICS_NAME[CAP] = {"HANG",
                                                           "START_DUMP",
                                                           "END_DUMP",
                                                           "POOL_QUEUE_SIZE",
                                                           "WORK_QUEUE_SIZE",
                                                           "GC_COUNT",
                                                           "DATA_LOADER_COUNT"};
  };

  struct MemMetrics {
    static constexpr uint32_t KERNEL_TYPE = 2;
    static constexpr std::string_view BUCKET_NAME = "MEMORY_P";
    static constexpr std::string_view TYPE = "memory";
    static constexpr int CAP = 1;
    static constexpr std::string_view GAUGE_PREFIX = "XPU_TIMER_MEMORY_";
    static constexpr std::string_view METRICS_NAME[CAP] = {"COUNTER"};
  };

  static constexpr int MAX_CAP =
      std::max({CollMetrics::CAP, MatmulMetrics::CAP, CommonMetrics::CAP,
                MemMetrics::CAP});
  static constexpr int MAX_KERNEL_TYPE =
      std::max({CollMetrics::KERNEL_TYPE, MatmulMetrics::KERNEL_TYPE,
                MemMetrics::KERNEL_TYPE});
  static constexpr int ALL_KERNEL_TYPE = (2 << MAX_KERNEL_TYPE) - 1;
};

struct KernelTraceConstant {
 public:
  static constexpr int DEFAULT_TRACE_COUNT = 1000;
  static constexpr std::string_view DEFAULT_TRACE_DUMP_PATH = "/root/timeline";
};

}  // namespace constant

}  // namespace xpu_timer
