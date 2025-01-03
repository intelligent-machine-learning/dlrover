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

#include <iostream>
#include <string>
#include <string_view>

#include "xpu_timer/nvidia/nvidia_timer.h"

int main() {
  std::string name = "test";
  uint64_t problem_size = 2;
  xpu_timer::Labels label;
  label["operation"] = "matmul";
  const char* cstr = "mm";
  const std::string_view type(cstr);

  std::string ttt = "mm";
  if (type == ttt) {
    std::cout << "can compare" << std::endl;
  }

  xpu_timer::metrics::performance_fn pfn =
      xpu_timer::nvidia::NvidiaGpuTimer::mmPerformance;

  xpu_timer::metrics::bucket_fn bfn =
      xpu_timer::nvidia::NvidiaGpuTimer::collBucketFn;

  auto m = std::make_shared<xpu_timer::metrics::MatCommuMetrics>(
      label, problem_size, name, type, pfn, bfn);
  std::vector<uint64_t> aa{1, 2};
  m->pushMetrics(aa);

  std::vector<uint64_t> res_m = m->getBvarValue();
  std::cout << res_m[0] << std::endl;
  std::string bucket_name = m->computeBucket();
  std::cout << bucket_name << std::endl;
  std::cout << m->isTimeout() << std::endl;
  xpu_timer::server::BrpcMetrics brpc_metrics;
  m->getMetrics(&brpc_metrics);
  std::cout << brpc_metrics.DebugString() << std::endl;

  m->getPerformance(20);
  // std::cout << m->performance_ << std::endl;
  uint64_t latency = 2;
  double performace_coef = 1e6;
  double res = double(problem_size) / latency / performace_coef;
  std::cout << res << std::endl;

  auto m_throughtput =
      std::make_shared<xpu_timer::metrics::ThroughPutSumMetrics>(
          label, problem_size, bucket_name, type, pfn, bfn);
  m_throughtput->pushMetrics({1, 2});
  std::cout << m_throughtput->canPush() << std::endl;

  std::string name_mem = "test2";
  uint64_t problem_size_mem = 2;
  xpu_timer::Labels label_mem;
  label_mem["operation"] = "CudaMem";
  const char* cstr_mem = "memory";
  std::string_view type_mem(cstr);

  std::cout << "MemMetrics" << std::endl;

  auto mem = std::make_shared<xpu_timer::metrics::MemMetrics>(
      label_mem, problem_size_mem, name_mem, type_mem);
  std::vector<uint64_t> amem{1, 2};
  mem->pushMetrics(amem);
  std::vector<uint64_t> res_mem = mem->getBvarValue();
  std::cout << res_mem[0] << std::endl;
}
