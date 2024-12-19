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

#include "xpu_timer/common/metrics.h"

#include "xpu_timer/common/logging.h"

namespace xpu_timer {
namespace metrics {

BaseMetrics::BaseMetrics(Labels label, uint64_t problem_size, std::string name,
                         const std::string_view& type)
    : problem_size_(problem_size), name_(name), type_(std::string(type)) {
  label_ = config::BvarMetricsConfig::common_label;
  label_.insert(label.begin(), label.end());
  if (type == constant::Metrics::MatmulMetrics::TYPE) {
    gauge_prefix_ = std::string(constant::Metrics::MatmulMetrics::GAUGE_PREFIX);
  } else if (type == constant::Metrics::CollMetrics::TYPE) {
    gauge_prefix_ = std::string(constant::Metrics::CollMetrics::GAUGE_PREFIX);
  } else if (type == constant::Metrics::MemMetrics::TYPE) {
    gauge_prefix_ = std::string(constant::Metrics::MemMetrics::GAUGE_PREFIX);
  }
  push_interval_ = config::BvarMetricsConfig::push_interval;
}

bool BaseMetrics::canPush() {
  if (std::chrono::steady_clock::now() - start_ < push_interval_) {
    return false;
  }
  start_ = std::chrono::steady_clock::now();
  return true;
}

BaseMetrics::~BaseMetrics() {}

std::string BaseMetrics::getName() { return name_; }
std::string BaseMetrics::getGaugePrefix() { return gauge_prefix_; }
Labels BaseMetrics::getLabel() { return label_; }

TimeoutMixinBase::~TimeoutMixinBase() {}

TimeoutMixin::TimeoutMixin() {
  inner_counter_ = std::make_unique<bvar::Adder<uint64_t>>();
  timout_counter_ = std::make_unique<bvar::Window<bvar::Adder<uint64_t>>>(
      inner_counter_.get(), config::BvarMetricsConfig::timeout_window_second);
  timeoutAdd();  // add 1 to avoid delete metric which is created but not push
}

ThroughPutMetrics::ThroughPutMetrics(Labels label, uint64_t problem_size,
                                     std::string name,
                                     const std::string_view& type)
    : BaseMetrics(label, problem_size, name, type) {
  latency_recorder_ = std::make_unique<bvar::LatencyRecorder>(
      config::BvarMetricsConfig::bvar_window_size);
  push_interval_ = config::BvarMetricsConfig::local_push_interval;
}
void ThroughPutMetrics::pushMetrics(const std::vector<uint64_t>& value) {
  *latency_recorder_ << value[0];  // push latency to latency_recorder_
}

std::vector<uint64_t> ThroughPutMetrics::getBvarValue() {
  uint64_t min_latency = latency_recorder_->min_latency();
  uint64_t avg_latency = latency_recorder_->latency();
  uint64_t p99_latency = latency_recorder_->latency_percentile(0.99);
  uint64_t max_latency = latency_recorder_->max_latency();
  return {avg_latency, min_latency, p99_latency, max_latency};
}

void ThroughPutMetrics::getMetrics(BrpcMetrics* request) {
  xpu_timer::server::KernelBrpcMetrics* metrics =
      request->mutable_kernel_metrics();
  metrics->set_avg_latency(latency_recorder_->latency());
  request->set_name(name_);
  request->set_local_rank(config::GlobalConfig::local_rank);
}

void MatCommuMetrics::pushMetrics(const std::vector<uint64_t>& value) {
  TimeoutMixin::timeoutAdd();
  ThroughPutMetrics::pushMetrics(value);
  MatCommuMetrics::getPerformance(/*problem_size*/ value[0]);
}

std::vector<uint64_t> MatCommuMetrics::getBvarValue() {
  std::vector<uint64_t> value = ThroughPutMetrics::getBvarValue();
  // {latency_us, problem_size}
  return {value[0], problem_size_};
}

MatCommuMetrics::MatCommuMetrics(Labels label, uint64_t problem_size,
                                 std::string name, const std::string_view& type,
                                 performance_fn pfn, bucket_fn bfn)
    : BucketingMixin(bfn),
      PerformanceMixin(pfn),
      ThroughPutMetrics(label, problem_size, name, type) {}

std::string MatCommuMetrics::computeBucket() {
  return BucketingMixin::computeBucket(performance_, problem_size_, &label_);
}

void MatCommuMetrics::getPerformance(uint64_t latency_in_us) {
  performance_ =
      PerformanceMixin::computePerformance(latency_in_us, problem_size_);
}

ThroughPutSumMetrics::ThroughPutSumMetrics(Labels label, uint64_t problem_size,
                                           std::string name,
                                           const std::string_view& type,
                                           performance_fn pfn, bucket_fn bfn)
    : MatCommuMetrics(label, problem_size, name, type, pfn, bfn) {
  push_interval_ = config::BvarMetricsConfig::push_interval;
}

bool TimeoutMixin::isTimeout() { return timout_counter_->get_value() == 0; }

void TimeoutMixin::timeoutAdd() { *inner_counter_ << 1; }

BucketingMixinBase::~BucketingMixinBase() {}
PerformanceMixinBase::~PerformanceMixinBase() {}

void ThroughPutSumMetrics::getMetrics(BrpcMetrics* request) {
  std::vector<uint64_t> res = ThroughPutMetrics::getBvarValue();
  uint64_t sum_latency_us = sum_latency_in_us.reset();
  uint64_t problem_size = sum_problem_size.reset();
  double performance;
  if (!sum_latency_us) {
    performance = 0.0;
  } else {
    performance =
        PerformanceMixin::computePerformance(sum_latency_us, problem_size);
  }

  xpu_timer::server::KernelBrpcMetrics* metrics =
      request->mutable_kernel_metrics();
  metrics->set_avg_latency(res[0]);  // avg
  metrics->set_min_latency(res[1]);  // min
  metrics->set_p99_latency(res[2]);  // p99
  metrics->set_max_latency(res[3]);  // max
  metrics->set_performance(performance);
  request->set_name(name_);
  request->set_local_rank(config::GlobalConfig::local_rank);
}

void ThroughPutSumMetrics::pushMetrics(const std::vector<uint64_t>& value) {
  // value {latency, problem_size}
  TimeoutMixin::timeoutAdd();
  ThroughPutMetrics::pushMetrics(value);
  sum_latency_in_us << value[0];
  sum_problem_size << value[1];
}

CountingMetrics::CountingMetrics(Labels label, uint64_t problem_size,
                                 std::string name, const std::string_view& type)
    : BaseMetrics(label, problem_size, name, type) {
  counter_ = std::make_unique<bvar::Adder<uint64_t>>();
  push_interval_ = config::BvarMetricsConfig::push_interval;
}

void CountingMetrics::pushMetrics(const std::vector<uint64_t>& value) {
  // CountingMetrics only have 1 value
  *counter_ << value[0];
}

std::vector<uint64_t> CountingMetrics::getBvarValue() {
  return {counter_->get_value()};
}

MemMetrics::MemMetrics(Labels label, uint64_t problem_size, std::string name,
                       const std::string_view& type)
    : CountingMetrics(label, problem_size, name, type) {}

void MemMetrics::getMetrics(BrpcMetrics* request) {
  xpu_timer::server::MemBrpcMetrics* metrics = request->mutable_mem_metrics();
  std::vector<uint64_t> count = CountingMetrics::getBvarValue();
  metrics->set_counter(count[0]);
  request->set_name(name_);
  request->set_local_rank(config::GlobalConfig::local_rank);
}

void MemMetrics::pushMetrics(const std::vector<uint64_t>& value) {
  CountingMetrics::pushMetrics({1});
}

}  // namespace metrics
}  // namespace xpu_timer
