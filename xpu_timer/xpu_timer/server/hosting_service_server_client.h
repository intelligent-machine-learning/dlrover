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
#include <brpc/channel.h>
#include <brpc/server.h>
#include <butil/logging.h>
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/protos/hosting_service.pb.h"

namespace xpu_timer {
namespace server {
using namespace util;

class LocalPrometheusService {
 public:
  // family and gauge are not owned, prometheus use unique_ptr
  // to manage those objects, so we only ref it.
  // clang-format off
  /*
        All_familys_ and all_gauges_ are described below, those static
        variable hold all Family<Gauge> and Gauge, all created gauge are
        store in all_gauges_. Each Family create lots of gauge with different
        labels.

        Each LocalPrometheusService object only hold the refs of gauges
        for pushing metrics.

        When a metrics is out of date, we remove the kernel name in all_gauges_,
        deregister metrics via the refs in the object.

                      ┌─────────────────────┐       ┌──────────────┐
                      │ATORCH_MM            ├──────►│prometheus::  │
        all_familys_  ├─────────────────────┤       │Family<Gauge> ├──────┐
                      │ATORCH_NCCL_KERNEL   │       ├──────────────┤      │
                      └─────────────────────┘       │QPS           │      │
                                                    ├──────────────┤      │
                                                    │AVG LATENCY   │      │
                                                    ├──────────────┤  create gauge
                                                    │MAX LATENCY   │      │
                                                    ├──────────────┤      │
                                                    │P99 LATENCY   │      │
                                                    ├──────────────┤      │
                                                    │P9999 LATENCY │      │
                                                    ├──────────────┤      │
                                                    │FLOPS         │      │
                    all_gauges_                     ├──────────────┤      │
                                                    │COUNT         │      │
          vector with size of local world size      └──────────────┘      │
       ─────────────────────────────────────────────                      │
      /         each rank data                      \                     │
     ┌──────────────────────────────────────┐       ┌──────────────┐      │
     │xpu_timer_mm_bmnk_1_1024_3072_1024_      ├──────►│prometheus::  │      │
     ├──────────────────────────────────────┤       │Gauge[7]      │◄─────┘
     │xpu_timer_nccl_kernel_all_gather_ringL   │       ├──────────────┤
     └──────────────────────────────────────┘       │QPS           │
                                                    ├──────────────┤
                                                    │AVG LATENCY   │
                                                    ├──────────────┤
                                                    │MAX LATENCY   │
                                                    ├──────────────┤
                                                    │P99 LATENCY   │
                                                    ├──────────────┤
                                                    │P9999 LATENCY │
                                                    ├──────────────┤
                                                    │FLOPS         │
                                                    ├──────────────┤
                                                    │COUNT         │
                                                    └──────────────┘
  */
  // clang-format on
  using Family_t = prometheus::Family<prometheus::Gauge>*;
  using Family_t_Array = std::array<Family_t, constant::Metrics::MAX_CAP>;
  using Gauge_t = prometheus::Gauge*;
  using Gauge_t_Array = std::array<Gauge_t, constant::Metrics::MAX_CAP>;
  using Gauge_t_MapArray = std::unordered_map<std::string, Gauge_t_Array>;

  static void setUp(int port, int local_world_size);
  void push(const BrpcMetrics* metrics) noexcept;
  LocalPrometheusService(const std::string& gauge_prefix,
                         const std::string& kernel_name,
                         const std::map<std::string, std::string>& label,
                         int rank) noexcept;
  LocalPrometheusService(LocalPrometheusService&& other) noexcept;
  ~LocalPrometheusService();

 private:
  Gauge_t_Array* gauges_;  // only ref, used in push and ~LocalPrometheusService
  int rank_;
  std::string kernel_name_;
  std::string gauge_prefix_;
  int max_cap_;
  static std::shared_ptr<prometheus::Registry> registry_;
  static prometheus::Exposer* exposer_;
  static butil::Mutex mu_;
  // type as key, ATORCH_NCCL_KERNEL_ or ATORCH_MM_KERNEL_
  static std::unordered_map<std::string, Family_t_Array> all_familys_;
  // kernel name as key, xpu_timer_mm_bmnk_1_1024_3072_1024_
  static std::vector<Gauge_t_MapArray> all_gauges_;

  template <typename MetricsType>
  static void setUpInner() {
    std::string prefix_name = std::string(MetricsType::GAUGE_PREFIX);
    auto& familys = all_familys_[prefix_name];
    LOG(INFO) << "Register Gauge Family " << prefix_name;
    for (int i = 0; i < MetricsType::CAP; i++) {
      std::string family_name =
          prefix_name + std::string(MetricsType::METRICS_NAME[i]);
      LOG(INFO) << "family " << family_name << " at " << i;
      auto& family =
          prometheus::BuildGauge().Name(family_name).Register(*registry_);
      familys[i] = &family;
    }
  }
};

class AsyncJob {
 public:
  brpc::Controller* cntl;
  google::protobuf::Closure* done;
  virtual void run() = 0;
  virtual ~AsyncJob() {};

  virtual void run_and_delete() {
    run();
    delete this;
  }

  template <typename T>
  static void* RunServerJob(void* args) {
    T* job = static_cast<T*>(args);
    job->run_and_delete();
    return nullptr;
  }
};

class StringStacktraceJob : public AsyncJob {
 public:
  void run() override;
  const StacktraceRequest* request;
  StacktraceResponse* response;

 private:
  butil::IOBuf stdout_buf;
  butil::IOBuf stderr_buf;
  virtual ~StringStacktraceJob() {};
};

class MainServer {
 public:
  static constexpr std::string_view DUMMY_SERVER = "DUMMY_SERVER";
  static constexpr std::string_view LOCAL_RANK_0_SERVER = "LOCAL_RANK_0_SERVER";
  static constexpr std::string_view HOSTING_SERVICE = "HOSTING_SERVICE";
  void start(const std::string_view& server_type);
  explicit MainServer(const std::string& endpoint, int thread_num,
                      int prometheus_port, int local_world_size);
  void join();

 private:
  brpc::Server server_;
  brpc::ServerOptions options_;
  std::string endpoint_;
  int thread_num_;
  int local_world_size_;
  std::unordered_map<std::string_view, google::protobuf::Service*> services_;
  void addService(const std::string_view& service_name);
};

class ClientStub {
 public:
  ClientStub(std::string endpoint);
  void requestDump(bool sync, int pid, int rank, int world_size,
                   const std::string& dump_path,
                   const std::vector<std::string>& hang_kernel);
  void requestRegisterPrometheus(
      bool sync, const std::string& name, const std::string& gauge_name,
      int rank, int local_rank,
      const std::map<std::string, std::string>& labels);
  void pushPrometheusCommonMetrics(
      bool sync, const std::string& name, int64_t local_rank, int64_t hang = -1,
      int64_t start_dump = -1, int64_t end_dump = -1,
      int64_t pool_queue_size = -1, int64_t work_queue_size = -1,
      int64_t gc_count = -1, int64_t data_loader_count = -1);
  void pushPrometheusThroughputMetrics(bool sync, const std::string& name,
                                       int local_rank, uint64_t avg_latency,
                                       uint64_t min_latency,
                                       uint64_t max_latency,
                                       uint64_t p99_latency,
                                       double performance);
  void pushPrometheusMemMetrics(bool sync, const std::string& name,
                                int local_rank, uint64_t counter);

  void requestDeRegisterPrometheus(bool sync, const std::string& name, int rank,
                                   int local_rank);
  void pushPrometheusMetrics(bool sync, const BrpcMetrics* request);
  void pushSignalFrameInfo(const SignalFrameRequest* request);

 private:
  brpc::Channel* channel_;
  brpc::ChannelOptions* options_;
  HostingService_Stub* stub_;

  static void HandleDumpResponse(brpc::Controller* cntl,
                                 StacktraceResponse* response);
  static void HandleRegisterPrometheusResponse(
      brpc::Controller* cntl, RegisterPrometheusResponse* response);
  static void HandlePushPrometheusResponse(brpc::Controller* cntl,
                                           google::protobuf::Empty* response);
  static void HandleDeregisterPrometheusResponse(
      brpc::Controller* cntl, RegisterPrometheusResponse* response);

  static void HandlePushSignalFrameInfo(brpc::Controller* cntl,
                                        google::protobuf::Empty* response,
                                        int signal, int rank);
};

class HostingServiceImpl : public HostingService {
 public:
  HostingServiceImpl(int local_world_size);
  void DumpStringStacktrace(google::protobuf::RpcController* cntl_base,
                            const StacktraceRequest* request,
                            StacktraceResponse* response,
                            google::protobuf::Closure* done);

  void RegisterPrometheus(google::protobuf::RpcController* cntl_base,
                          const RegisterPrometheusRequest* request,
                          RegisterPrometheusResponse* response,
                          google::protobuf::Closure* done);

  void PushPrometheus(google::protobuf::RpcController* cntl_base,
                      const BrpcMetrics* request,
                      google::protobuf::Empty* response,
                      google::protobuf::Closure* done);

  void DumpKernelTrace(google::protobuf::RpcController* cntl_base,
                       const DumpKernelTraceRequest* request,
                       DumpKernelTraceResponse* response,
                       google::protobuf::Closure* done);

  void DeRegisterPrometheus(google::protobuf::RpcController* cntl_base,
                            const DeRegisterPrometheusRequest* request,
                            RegisterPrometheusResponse* response,
                            google::protobuf::Closure* done);

  void PushSignalFrameInfo(google::protobuf::RpcController* cntl_base,
                           const SignalFrameRequest* request,
                           google::protobuf::Empty* response,
                           google::protobuf::Closure* done);

 private:
  std::vector<
      std::unordered_map<std::string, std::unique_ptr<LocalPrometheusService>>>
      prometheus_services_;
  int local_world_size_;
  int rank_;
  std::unique_ptr<util::ShmSwitch> switch_;
  std::vector<std::unique_ptr<butil::Mutex>> mus_;
  std::unique_ptr<butil::Mutex> mu_;
};
}  // namespace server
}  // namespace xpu_timer
