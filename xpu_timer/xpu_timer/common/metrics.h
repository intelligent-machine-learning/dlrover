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
#include <bvar/bvar.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/platform.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/protos/hosting_service.pb.h"

namespace xpu_timer {
namespace metrics {

using BrpcMetrics = xpu_timer::server::BrpcMetrics;
using performance_fn = std::function<double(uint64_t, uint64_t)>;
using bucket_fn = std::function<std::string(double, uint64_t, Labels*)>;

// there three kind of metrics, i.e. CommonMetrics, MatCollMetrics, MemMemtrics
struct CommonMetrics {
  explicit CommonMetrics() {
    hang = start_dump = end_dump = pool_queue_size = work_queue_size =
        gc_count = data_loader_count = -1;
  }
  int64_t hang;
  int64_t start_dump;
  int64_t end_dump;
  int64_t pool_queue_size;
  int64_t work_queue_size;
  int64_t gc_count;
  int64_t data_loader_count;
};

class BaseMetrics {
 public:
  virtual void pushMetrics(const std::vector<uint64_t>& value) = 0;
  virtual void getMetrics(BrpcMetrics* request) = 0;
  virtual ~BaseMetrics() = 0;
  BaseMetrics(Labels label, uint64_t problem_size, std::string name,
              const std::string_view& type);

  bool canPush();
  std::string getName();
  std::string getGaugePrefix();
  Labels getLabel();

 protected:
  Labels label_;
  uint64_t problem_size_;
  std::string name_;
  std::string type_;
  std::chrono::steady_clock::time_point start_;
  std::string gauge_prefix_;
  std::chrono::seconds push_interval_;
};

class TimeoutMixinBase {
 public:
  virtual bool isTimeout() = 0;
  virtual ~TimeoutMixinBase() = 0;
};

class TimeoutMixin : public TimeoutMixinBase {
 public:
  explicit TimeoutMixin();
  bool isTimeout() override;
  virtual ~TimeoutMixin() {};

 protected:
  void timeoutAdd();

 private:
  std::unique_ptr<bvar::Adder<uint64_t>> inner_counter_;
  std::unique_ptr<bvar::Window<bvar::Adder<uint64_t>>> timout_counter_;
};

class ThroughPutMetrics : public BaseMetrics {
 public:
  ThroughPutMetrics(Labels label, uint64_t problem_size, std::string name,
                    const std::string_view& type);
  void pushMetrics(const std::vector<uint64_t>& value);
  virtual ~ThroughPutMetrics() {};

  std::vector<uint64_t> getBvarValue();
  void getMetrics(BrpcMetrics* request);

 protected:
  std::unique_ptr<bvar::LatencyRecorder> latency_recorder_;
};

class BucketingMixinBase {
 public:
  virtual std::string computeBucket(double performance, uint64_t problem_size,
                                    Labels* label) = 0;
  virtual ~BucketingMixinBase() = 0;
};

class BucketingMixin : public BucketingMixinBase {
 public:
  BucketingMixin(bucket_fn func) : func_(func) {}
  virtual ~BucketingMixin() {};
  std::string computeBucket(double performance, uint64_t problem_size,
                            Labels* label) override {
    return func_(performance, problem_size, label);
  }

 protected:
  bucket_fn func_;
};

class PerformanceMixinBase {
 public:
  virtual double computePerformance(uint64_t latency_in_us,
                                    uint64_t problem_size) = 0;
  virtual ~PerformanceMixinBase() = 0;
};

class PerformanceMixin : public PerformanceMixinBase {
 public:
  PerformanceMixin(performance_fn func) : func_(func) {}
  virtual ~PerformanceMixin() {}
  double computePerformance(uint64_t latency_in_us, uint64_t problem_size) {
    return func_(latency_in_us, problem_size);
  }

 protected:
  performance_fn func_;
};

class MatCommuMetrics : public BucketingMixin,
                        public PerformanceMixin,
                        public ThroughPutMetrics,
                        public TimeoutMixin {
 public:
  MatCommuMetrics(Labels label, uint64_t problem_size, std::string name,
                  const std::string_view& type, performance_fn pfn,
                  bucket_fn bfn);

  void pushMetrics(const std::vector<uint64_t>& value);
  std::vector<uint64_t> getBvarValue();
  std::string computeBucket();
  void getPerformance(uint64_t latency_in_us);
  virtual ~MatCommuMetrics() {};

 private:
  uint64_t performance_;
};

class ThroughPutSumMetrics : public MatCommuMetrics {
 public:
  ThroughPutSumMetrics(Labels label, uint64_t problem_size, std::string name,
                       const std::string_view& type, performance_fn pfn,
                       bucket_fn bfn);
  bvar::Adder<uint64_t> sum_problem_size;
  bvar::Adder<uint64_t> sum_latency_in_us;

  void pushMetrics(const std::vector<uint64_t>& value);
  void getMetrics(BrpcMetrics* request);
  ~ThroughPutSumMetrics() {};
};

class CountingMetrics : public BaseMetrics {
 public:
  CountingMetrics(Labels label, uint64_t problem_size, std::string name,
                  const std::string_view& type);
  std::vector<uint64_t> getBvarValue();
  void pushMetrics(const std::vector<uint64_t>& value) override;
  virtual ~CountingMetrics() {};

 protected:
  std::unique_ptr<bvar::Adder<uint64_t>> counter_;
};

class MemMetrics : public CountingMetrics {
 public:
  MemMetrics(Labels label, uint64_t problem_size, std::string name,
             const std::string_view& type);

  void pushMetrics(const std::vector<uint64_t>& value);
  void getMetrics(BrpcMetrics* request);
  ~MemMetrics() {};
};

}  // end of namespace metrics
}  // namespace xpu_timer
