#pragma once
#include <algorithm>

namespace xpu_timer {
namespace constant {

static constexpr bool SKIP_TP = true;

static constexpr uint64_t US_TO_NS = 1000000;

struct Metrics {
 public:
  static constexpr int AVG_LATENCY = 0;
  static constexpr int MAX_LATENCY = 1;
  static constexpr int P99_LATENCY = 2;
  static constexpr int MIN_LATENCY = 3;
  static constexpr int PERFORMANCE = 4;

  static constexpr int COUNTER = 0;
  static constexpr int GAUGE = 1;

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
    static constexpr int CAP = 1;
    static constexpr std::string_view METRICS_NAME[CAP] = {"METRICS"};
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

enum class CollType {
  Send = 0,  // use 0 and 1 to judge type of Send and Recv in SendRecv
  Recv = 1,
  AllReduce,
  Reduce,
  AllGather,
  ReduceScatter,
  Broadcast,
};

};  // namespace constant

}  // namespace xpu_timer
