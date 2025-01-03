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

#include "xpu_timer/common/bvar_prometheus.h"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/server/hosting_service_server_client.h"

namespace xpu_timer {
using namespace util;

/*
 * ===================================
 * MetricsManager variables and public functions
 * ===================================
 */

MetricsManager::MetricsManager(std::shared_ptr<server::ClientStub> client_stub)
    : should_run_(true) {
  deregister_thread_ = std::thread(&MetricsManager::checkMetrics, this);
#ifdef _GNU_SOURCE
  // pthread_setname_np is not posix
  auto handle = deregister_thread_.native_handle();
  pthread_setname_np(handle, "xpu_bvar");
#endif
  deregister_timeout_count_ =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_DEREGISTER_COUNT");
  client_stub_ = client_stub;
};

MetricsManager::~MetricsManager() {
  should_run_.store(false);
  XLOG(INFO) << "Delete MetricsManager";

  quit_checking_thread_cv_.notify_one();
  if (deregister_thread_.joinable()) {
    deregister_thread_.join();
  }
  XLOG(INFO) << "Checking thread is stopped";

  for (auto it : timeout_metrics_) {
    it.second.reset();
  }

  for (auto it : mem_metrics_) {
    it.second.reset();
  }

  for (auto it : throughput_sum_metrics_) {
    it.second.reset();
  }

  for (auto it : mat_comm_metrics_) {
    it.second.reset();
  }
}

void MetricsManager::updateMetrics(XpuTimer* work_item,
                                   metrics::performance_fn pfn,
                                   metrics::bucket_fn bfn) {
  // first get item's name and type
  std::lock_guard<std::mutex> lock(mu_);
  const std::string_view& type = work_item->getType();
  if (type == constant::Metrics::MatmulMetrics::TYPE ||
      type == constant::Metrics::CollMetrics::TYPE) {
    updateMatCommuMetrics(work_item, pfn, bfn);
  } else if (type == constant::Metrics::MemMetrics::TYPE) {
    updateMemMetrics(work_item);
  }
}

void MetricsManager::updateMemMetrics(XpuTimer* work_item) {
  const std::string& name = work_item->getName();
  auto it = mem_metrics_.find(name);
  if (it != mem_metrics_.end()) {
    it->second->pushMetrics({1});  // counter
    if (it->second->canPush()) {
      pushMemMetricsToRemote(it->second);
    }
  } else {
    auto m = std::make_shared<metrics::MemMetrics>(
        work_item->getExtraLabels(), work_item->getProblemSize(),
        work_item->getName(), work_item->getType());
    mem_metrics_[name] = m;
    m->pushMetrics({1});
    registerMetrics(m->getName(), m->getGaugePrefix(), m->getLabel());
  }
}

void MetricsManager::updateMatCommuMetrics(XpuTimer* work_item,
                                           metrics::performance_fn pfn,
                                           metrics::bucket_fn bfn) {
  const std::string& name = work_item->getName();
  std::uint64_t duration_us = work_item->getDuration();

  auto it = mat_comm_metrics_.find(name);
  // if the metric exist, push duration to it
  if (it != mat_comm_metrics_.end()) {
    it->second->pushMetrics({duration_us});
    if (!it->second->canPush()) return;
    const std::string& bucket_name = it->second->computeBucket();
    auto th_it = throughput_sum_metrics_.find(bucket_name);
    if (th_it != throughput_sum_metrics_.end()) {
      std::vector<uint64_t> latency_problem_size = it->second->getBvarValue();
      th_it->second->pushMetrics(
          latency_problem_size);  // push duration, problem size

      if (!th_it->second->canPush()) return;
      pushThroughtPutSumMetricsToRemote(th_it->second);
    } else {
      auto m = std::make_shared<metrics::ThroughPutSumMetrics>(
          it->second->getLabel(), work_item->getProblemSize(), bucket_name,
          work_item->getType(), pfn, bfn);
      throughput_sum_metrics_[bucket_name] = m;
      timeout_metrics_[bucket_name] = m;
      registerMetrics(m->getName(), m->getGaugePrefix(), m->getLabel());
    }
  } else {
    auto mc = std::make_shared<metrics::MatCommuMetrics>(
        work_item->getExtraLabels(), work_item->getProblemSize(),
        work_item->getName(), work_item->getType(), pfn, bfn);
    mc->pushMetrics({duration_us});
    timeout_metrics_[name] = mc;
    mat_comm_metrics_[name] = mc;
  }
}
void MetricsManager::registerMetrics(std::string name, std::string gauge_prefix,
                                     Labels label) {
  client_stub_->requestRegisterPrometheus(
      true, name, gauge_prefix, config::GlobalConfig::rank,
      config::GlobalConfig::local_rank, label);
}

void MetricsManager::deregisterMetrics() {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> names;
  std::vector<std::string> metrics_names;
  for (auto& it : timeout_metrics_) {
    if (it.second->isTimeout()) {
      names.push_back(it.first);
    }
  }
  for (const auto& name : names) {
    // delete timeout_metrics_[name];
    // common_metrics and mem_metrics never timeout
    auto it = throughput_sum_metrics_.find(name);
    if (it != throughput_sum_metrics_.end()) {
      metrics_names.push_back(it->second->getName());
    }
    timeout_metrics_.erase(name);
    mat_comm_metrics_.erase(name);
    throughput_sum_metrics_.erase(name);
  }
  for (const auto& name : metrics_names) {
    XLOG(INFO) << "delete " << name;
    client_stub_->requestDeRegisterPrometheus(false, name,
                                              config::GlobalConfig::rank,
                                              config::GlobalConfig::local_rank);
  }
}

void MetricsManager::checkMetrics() {
  int deregister_check_interval =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_DEREGISTER_TIME");
  while (should_run_.load()) {
    std::unique_lock<std::mutex> l(quit_checking_thread_mu_);
    quit_checking_thread_cv_.wait_for(
        l, std::chrono::seconds(deregister_check_interval),
        [this] { return !should_run_.load(); });
    deregisterMetrics();
  }
}

void MetricsManager::pushCommonMetricsToRemote(
    const std::string& metric_name,
    const metrics::CommonMetrics& comm_metrics) {
  auto it = common_metrics_.find(metric_name);
  if (it == common_metrics_.end()) {
    client_stub_->requestRegisterPrometheus(
        true, metric_name,
        std::string(constant::Metrics::CommonMetrics::GAUGE_PREFIX),
        config::GlobalConfig::rank, config::GlobalConfig::local_rank,
        config::BvarMetricsConfig::common_label);
    common_metrics_.insert(metric_name);
  }
  client_stub_->pushPrometheusCommonMetrics(
      false, metric_name, config::GlobalConfig::local_rank, comm_metrics.hang,
      comm_metrics.start_dump, comm_metrics.end_dump,
      comm_metrics.pool_queue_size, comm_metrics.work_queue_size,
      comm_metrics.gc_count, comm_metrics.data_loader_count);
}

void MetricsManager::pushMemMetricsToRemote(
    std::shared_ptr<metrics::MemMetrics> t) {
  const std::string& name = t->getName();
  metrics::BrpcMetrics request;
  t->getMetrics(&request);
  client_stub_->pushPrometheusMetrics(false, &request);
}

void MetricsManager::pushThroughtPutSumMetricsToRemote(
    std::shared_ptr<metrics::ThroughPutSumMetrics> t) {
  const std::string& name = t->getName();
  metrics::BrpcMetrics request;
  t->getMetrics(&request);
  client_stub_->pushPrometheusMetrics(false, &request);
}
}  // namespace xpu_timer
