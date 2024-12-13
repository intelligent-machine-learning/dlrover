#pragma once
#include <bvar/bvar.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "xpu_timer/common/metrics.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/common/xpu_timer.h"

namespace xpu_timer {
using namespace util;
namespace server {
class ClientStub;
}

class MetricsManager {
 public:
  // Manage metrics. There are two types of metrics.
  // 	1. kernel metric
  // 	2. bucketing metric
  // Each type of metric has a hashmap, key is metric name.
  // For kernel metric, name is passed from XpuTimer object.
  // For bucketing metric, name is type and level, such as COLL_P10, MM_P4.
  // metrics is managed by kernel name, and use rpc
  // to register metrics.
  MetricsManager(std::shared_ptr<server::ClientStub> client_stub);

  ~MetricsManager();

  void updateMetrics(XpuTimer* work_item, metrics::performance_fn pfn,
                     metrics::bucket_fn bfn);
  void DeleteMetrics(bool exit = false);
  void registerMetrics(std::string name, std::string gauge_prefix,
                       Labels label);
  void deregisterMetrics();
  void pushMetricsToRemote();
  void pushCommonMetricsToRemote(const std::string& metric_name,
                                 const metrics::CommonMetrics& params);
  void pushThroughtPutSumMetricsToRemote(
      std::shared_ptr<metrics::ThroughPutSumMetrics> t);
  void pushMemMetricsToRemote(std::shared_ptr<metrics::MemMetrics> t);
  void updateMatCommuMetrics(XpuTimer* work_item, metrics::performance_fn pfn,
                             metrics::bucket_fn bfn);
  void updateMemMetrics(XpuTimer* work_item);

 private:
  std::unordered_set<std::string> common_metrics_;
  std::unordered_map<std::string, std::shared_ptr<metrics::TimeoutMixinBase>>
      timeout_metrics_;
  std::unordered_map<std::string,
                     std::shared_ptr<metrics::ThroughPutSumMetrics>>
      throughput_sum_metrics_;
  std::unordered_map<std::string, std::shared_ptr<metrics::MatCommuMetrics>>
      mat_comm_metrics_;
  std::unordered_map<std::string, std::shared_ptr<metrics::MemMetrics>>
      mem_metrics_;

  std::shared_ptr<server::ClientStub> client_stub_;  // not owned
  std::atomic<bool> should_run_;
  std::thread deregister_thread_;
  std::mutex mu_;
  std::mutex quit_checking_thread_mu_;
  std::condition_variable quit_checking_thread_cv_;
  int deregister_timeout_count_;

  void checkMetrics();
};
}  // namespace xpu_timer
