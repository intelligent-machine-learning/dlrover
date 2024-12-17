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

#include "xpu_timer/server/hosting_service_server_client.h"

#include <brpc/server.h>
#include <google/protobuf/empty.pb.h>

#include <boost/process.hpp>
#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <thread>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/server/python_plugin.h"

namespace bp = ::boost::process;

namespace xpu_timer {
namespace server {
namespace detail {

// boot dump process
int callShellByBoost(const std::string& bin_path,
                     const std::initializer_list<std::string>& args,
                     butil::IOBuf* stdout_buf, butil::IOBuf* stderr_buf,
                     const std::string& hang_kernels) {
  bp::environment env = boost::this_process::environment();
  env.erase("LD_PRELOAD");
  bp::ipstream out_stream;
  bp::ipstream err_stream;

  std::string bin = bp::search_path(bin_path).string();
  if (bin.empty()) {
    stderr_buf->append(bin_path);
    stderr_buf->append("is not found");
    return 1;
  }

  bp::child c(bin, bp::args(args), env, bp::std_out > out_stream,
              bp::std_err > err_stream);
  std::string line;
  while (std::getline(out_stream, line)) {
    stdout_buf->append(line);
    stdout_buf->append("\n");
    LOG(INFO) << line;
  }
  while (std::getline(err_stream, line)) {
    stderr_buf->append(line);
    stderr_buf->append("\n");
    LOG(INFO) << line;
  }
  c.wait();
  int exit_code = c.exit_code();
  if (exit_code) {
    LOG(WARNING) << "Call " << bin << " error, code is " << exit_code;
  }
  return exit_code;
}
}  // namespace detail

void StringStacktraceJob::run() {
  brpc::ClosureGuard done_guard(done);
  std::string pid = std::to_string(request->pid());
  std::string rank = std::to_string(request->rank());
  std::string world_size = std::to_string(request->world_size());
  std::string serialized_hang_kernels;
  request->hang_kernels().SerializeToString(&serialized_hang_kernels);
  // xpu_timer_dump_driver --pid 1142 --rank 1 --world-size 2 --dump-path /root,
  // dump to /root
  int ret_code = detail::callShellByBoost(
      "xpu_timer_dump_driver",
      {"--pid", pid, "--rank", rank, "--world-size", world_size, "--dump-path",
       request->dump_path(), "--gdb", "--pyspy"},
      &stdout_buf, &stderr_buf, serialized_hang_kernels);
  if (ret_code) {
    LOG(WARNING) << "dump error " << ret_code << " " << stderr_buf.to_string();
  } else {
    LOG(INFO) << "dump " << stdout_buf.to_string();
    response->set_stacktrace(stdout_buf.to_string());
  }
}

/*
 * ===================================
 * LocalPrometheusService variables and public functions
 * ===================================
 */
std::shared_ptr<prometheus::Registry> LocalPrometheusService::registry_;
prometheus::Exposer* LocalPrometheusService::exposer_ = nullptr;
std::vector<LocalPrometheusService::Gauge_t_MapArray>
    LocalPrometheusService::all_gauges_;
std::unordered_map<std::string, LocalPrometheusService::Family_t_Array>
    LocalPrometheusService::all_familys_;
butil::Mutex LocalPrometheusService::mu_;

LocalPrometheusService::LocalPrometheusService(
    const std::string& gauge_prefix, const std::string& kernel_name,
    const std::map<std::string, std::string>& label, int rank) noexcept {
  rank_ = rank;
  kernel_name_ = kernel_name;
  gauge_prefix_ = gauge_prefix;

  BAIDU_SCOPED_LOCK(LocalPrometheusService::mu_);

  auto& all_familys = all_familys_[gauge_prefix];
  auto& all_gauges = all_gauges_[rank][kernel_name];
  gauges_ = &all_gauges;
  if (gauge_prefix_ == constant::Metrics::CollMetrics::GAUGE_PREFIX)
    max_cap_ = constant::Metrics::CollMetrics::CAP;
  else if (gauge_prefix_ == constant::Metrics::MatmulMetrics::GAUGE_PREFIX)
    max_cap_ = constant::Metrics::MatmulMetrics::CAP;
  else if (gauge_prefix_ == constant::Metrics::CommonMetrics::GAUGE_PREFIX)
    max_cap_ = constant::Metrics::CommonMetrics::CAP;
  else if (gauge_prefix_ == constant::Metrics::MemMetrics::GAUGE_PREFIX)
    max_cap_ = constant::Metrics::MemMetrics::CAP;

  for (int i = 0; i < max_cap_; i++) {
    auto& gauge = all_familys[i]->Add(label);
    gauges_->at(i) = &gauge;
  }
}
LocalPrometheusService::LocalPrometheusService(
    LocalPrometheusService&& other) noexcept {
  rank_ = other.rank_;
  gauges_ = std::move(other.gauges_);
  kernel_name_ = std::move(other.kernel_name_);
  gauge_prefix_ = std::move(other.gauge_prefix_);
  max_cap_ = other.max_cap_;
}

LocalPrometheusService::~LocalPrometheusService() {
  BAIDU_SCOPED_LOCK(LocalPrometheusService::mu_);
  auto& all_familys = all_familys_[gauge_prefix_];

  // delete key
  all_gauges_[rank_].erase(kernel_name_);
  // delete pointer
  for (int i = 0; i < max_cap_; i++) all_familys[i]->Remove(gauges_->at(i));
}

void LocalPrometheusService::setUp(int port, int local_world_size) {
  std::ostringstream oss;
  oss << "0.0.0.0:" << port;
  exposer_ = new prometheus::Exposer(oss.str());
  registry_ = std::make_shared<prometheus::Registry>();
  exposer_->RegisterCollectable(registry_);
  LOG(INFO) << "Start prometheus at 0.0.0.0:" << port;
  setUpInner<constant::Metrics::CollMetrics>();
  setUpInner<constant::Metrics::MatmulMetrics>();
  setUpInner<constant::Metrics::CommonMetrics>();
  setUpInner<constant::Metrics::MemMetrics>();

  all_gauges_.resize(local_world_size);
  all_gauges_.reserve(local_world_size);
}

void LocalPrometheusService::push(const BrpcMetrics* metrics) noexcept {
  if (metrics->has_kernel_metrics()) {
    const KernelBrpcMetrics& kernel_metrics = metrics->kernel_metrics();
    gauges_->at(constant::Metrics::AVG_LATENCY)
        ->Set(kernel_metrics.avg_latency());
    gauges_->at(constant::Metrics::MAX_LATENCY)
        ->Set(kernel_metrics.max_latency());
    gauges_->at(constant::Metrics::P99_LATENCY)
        ->Set(kernel_metrics.p99_latency());
    gauges_->at(constant::Metrics::MIN_LATENCY)
        ->Set(kernel_metrics.min_latency());
    gauges_->at(constant::Metrics::PERFORMANCE)
        ->Set(kernel_metrics.performance());
  } else if (metrics->has_common_metrics()) {
    const CommonBrpcMetrics& common_metrics = metrics->common_metrics();
    for (int i = 0; i < common_metrics.metrics_size(); ++i) {
      int64_t m = common_metrics.metrics(i);
      if (m != -1) gauges_->at(i)->Set(m);
    }
  } else if (metrics->has_mem_metrics()) {
    const MemBrpcMetrics& mem_metrics = metrics->mem_metrics();
    gauges_->at(constant::Metrics::COUNTER)->Set(mem_metrics.counter());
  }
}

/*
 * ===================================
 * HostingServiceImpl variables and public functions
 * ===================================
 */
HostingServiceImpl::HostingServiceImpl(int local_world_size)
    : local_world_size_(local_world_size) {
  prometheus_services_.resize(local_world_size);
  prometheus_services_.reserve(local_world_size);
  for (int i = 0; i < local_world_size; ++i) {
    mus_.push_back(std::make_unique<butil::Mutex>());
  }
  mu_ = std::make_unique<butil::Mutex>();
  // local_world_size_+1 means world size will include this daemon server
  switch_ = std::make_unique<util::ShmSwitch>(local_world_size_ + 1,
                                              local_world_size_, true);
}
void HostingServiceImpl::DumpStringStacktrace(
    google::protobuf::RpcController* cntl_base,
    const StacktraceRequest* request, StacktraceResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);

  StringStacktraceJob* job = new StringStacktraceJob;
  job->cntl = cntl;
  job->request = request;
  job->response = response;
  job->done = done;
  bthread_t th;
  LOG(INFO) << "Starting dump stack for rank " << request->rank();
  CHECK_EQ(0,
           bthread_start_background(
               &th, nullptr, AsyncJob::RunServerJob<StringStacktraceJob>, job));
  bthread_join(th, nullptr);
  LOG(INFO) << "response " << response;
  done_guard.release();
}

void HostingServiceImpl::RegisterPrometheus(
    google::protobuf::RpcController* cntl_base,
    const RegisterPrometheusRequest* request,
    RegisterPrometheusResponse* response, google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  const std::string& kernel_name = request->kernel_name();
  const std::string& gauge_prefix = request->gauge_prefix();
  // caller ensure each name only call once
  std::map<std::string, std::string> prometheus_labels;
  int rank = request->rank();
  int local_rank = request->local_rank();
  for (const auto& pair : request->labels())
    prometheus_labels[pair.first] = pair.second;

  if (local_rank < local_world_size_) {  // only for safe, always true
    BAIDU_SCOPED_LOCK(*mus_[local_rank]);
    prometheus_services_[local_rank].insert(
        {kernel_name,
         std::make_unique<LocalPrometheusService>(
             gauge_prefix, kernel_name, prometheus_labels, local_rank)});
  }

  LOG(INFO) << "Register name " << kernel_name << " gauge " << gauge_prefix
            << " local_rank " << local_rank << " rank " << rank;
  response->set_name(gauge_prefix + kernel_name);
  response->set_ret_code(0);
}

void HostingServiceImpl::PushPrometheus(
    google::protobuf::RpcController* cntl_base, const BrpcMetrics* request,
    google::protobuf::Empty* response, google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  int local_rank = request->local_rank();
  const std::string& name = request->name();

  if (local_rank < local_world_size_) {  // only for safe, always true
    BAIDU_SCOPED_LOCK(*mus_[local_rank]);
    std::unordered_map<std::string, std::unique_ptr<LocalPrometheusService>>&
        m = prometheus_services_[local_rank];
    if (auto it = m.find(name); it != m.end()) {
      m[name]->push(request);
    }
  }
}

void HostingServiceImpl::DumpKernelTrace(
    google::protobuf::RpcController* cntl_base,
    const DumpKernelTraceRequest* request, DumpKernelTraceResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  const std::string& dump_path = request->dump_path();
  uint32_t dump_count = request->dump_count();
  uint32_t dump_kernel_type = request->dump_kernel_type();
  if (dump_kernel_type == 0)
    dump_kernel_type = constant::Metrics::ALL_KERNEL_TYPE;
  uint64_t dump_time = request->dump_time();
  bool reset = request->reset();
  response->set_node(config::GlobalConfig::pod_name);
  if (switch_->getObj()->reset_flag) {
    LOG(INFO) << "Resetting status, ignore";
    response->set_msg("Resetting status, ignore");
    return;
  }
  if (reset) {
    LOG(INFO) << "Force reset dumping status";
    switch_->getObj()->start_dump = 0;
    switch_->getObj()->reset_flag = true;
    response->set_msg("Force reset dumping status");
    return;
  } else if (switch_->getObj()->start_dump) {
    LOG(INFO) << "The process is dumping, skip";
    response->set_msg("The process is dumping, skip");
    return;
  }
  // oss
  const DumpKernelTraceRequest::OssArgs& oss_dump_args = request->oss_args();
  std::string oss_dump_args_str;
  oss_dump_args.SerializeToString(&oss_dump_args_str);
  switch_->getObj()->reset(dump_path, oss_dump_args_str, dump_count, dump_time,
                           dump_kernel_type);
  switch_->getObj()->start_dump = 1;
  response->set_msg("Starting dumping");
  LOG(INFO) << "Dumping to " << dump_path << " with " << dump_count
            << " events upload to " << oss_dump_args.oss_path();
}

void HostingServiceImpl::DeRegisterPrometheus(
    google::protobuf::RpcController* cntl_base,
    const DeRegisterPrometheusRequest* request,
    RegisterPrometheusResponse* response, google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  const std::string& name = request->name();
  int local_rank = request->local_rank();
  int rank = request->rank();
  if (local_rank < local_world_size_) {
    BAIDU_SCOPED_LOCK(*mus_[local_rank]);
    std::unordered_map<std::string, std::unique_ptr<LocalPrometheusService>>&
        m = prometheus_services_[local_rank];
    if (auto it = m.find(name); it != m.end()) {
      LOG(INFO) << "DeRegister name " << name << " rank " << rank
                << " local_rank " << local_rank;
      m.erase(name);
    }
    response->set_ret_code(0);
  }
}
void HostingServiceImpl::PushSignalFrameInfo(
    google::protobuf::RpcController* cntl_base,
    const SignalFrameRequest* request, google::protobuf::Empty* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  // We need to held gil lock, this is conflict with bthread, so we add
  // -usercode_in_pthread in xpu_timer_daemon
  SignalFrameRequest new_request{*request};
  std::thread([new_request]() { runPythonPlugin(&new_request); }).detach();
}

/*
 * ===================================
 * MainServer variables and public functions
 * ===================================
 */
MainServer::MainServer(const std::string& endpoint, int thread_num,
                       int prometheus_port, int local_world_size)
    : endpoint_(endpoint),
      thread_num_(thread_num),
      local_world_size_(local_world_size) {
  LocalPrometheusService::setUp(prometheus_port, local_world_size);
  // thread_num is local world size
  if (thread_num == 0) thread_num_ = 1;
  thread_num_ = thread_num_ * 10;
}

void MainServer::join() { server_.RunUntilAskedToQuit(); }

void MainServer::addService(const std::string_view& service_name) {
  if (service_name == MainServer::HOSTING_SERVICE) {
    if (auto it = services_.find(service_name); it == services_.end()) {
      HostingServiceImpl* hosting_service_impl =
          new HostingServiceImpl(local_world_size_);
      if (server_.AddService(hosting_service_impl,
                             brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "Fail to add service " << service_name;
        return;
      }
      LOG(INFO) << "Adding service " << service_name;
      services_[service_name] = hosting_service_impl;
    } else {
      LOG(INFO) << "Service " << service_name << " has been added, skiping";
    }
  }
}

void MainServer::start(const std::string_view& server_type) {
  if (server_type == MainServer::DUMMY_SERVER) {
    options_.num_threads = 0;
  } else if (server_type == MainServer::LOCAL_RANK_0_SERVER) {
    options_.num_threads = thread_num_;
    addService(MainServer::HOSTING_SERVICE);
  }
  if (server_.Start(endpoint_.c_str(), &options_) != 0) {
    LOG(ERROR) << "Fail to start dummy_server at " << endpoint_;
    return;
  }
  LOG(INFO) << "Start server at " << endpoint_.c_str() << " with threads "
            << thread_num_;
}

/*
 * ===================================
 * ClientStub variables and public functions
 * ===================================
 */
ClientStub::ClientStub(std::string endpoint) {
  channel_ = new brpc::Channel();
  options_ = new brpc::ChannelOptions();
  options_->connection_type = "pooled";
  options_->timeout_ms = 100 * 1000UL;

  while (channel_->Init(endpoint.c_str(), "", options_) != 0) {
    XLOG(ERROR) << "Fail to initialize channel " << endpoint
                << " wait 3 seconds";
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  stub_ = new HostingService_Stub(channel_);
}

void ClientStub::HandleDumpResponse(brpc::Controller* cntl,
                                    StacktraceResponse* response) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<StacktraceResponse> response_guard(response);

  if (cntl->Failed()) {
    XLOG(WARNING) << "Fail to send StacktraceRequest, " << cntl->ErrorText();
    return;
  }
  XLOG(INFO) << "Received response from " << cntl->remote_side()
             << " stacktrace\n"
             << response->stacktrace();
}

void ClientStub::requestDump(bool sync, int pid, int rank, int world_size,
                             const std::string& dump_path,
                             const std::vector<std::string>& hang_kernels) {
  StacktraceRequest request;
  request.set_pid(pid);
  request.set_rank(rank);
  request.set_world_size(world_size);
  request.set_dump_path(dump_path);
  for (const auto& hang_kernel : hang_kernels)
    request.mutable_hang_kernels()->add_hang_kernels(hang_kernel);
  StacktraceResponse* response = new StacktraceResponse();
  brpc::Controller* cntl = new brpc::Controller();
  if (sync) {
    stub_->DumpStringStacktrace(cntl, &request, response, NULL);
    ClientStub::HandleDumpResponse(cntl, response);
  } else {
    google::protobuf::Closure* done =
        brpc::NewCallback(&ClientStub::HandleDumpResponse, cntl, response);
    stub_->DumpStringStacktrace(cntl, &request, response, done);
  }
}

void ClientStub::HandleRegisterPrometheusResponse(
    brpc::Controller* cntl, RegisterPrometheusResponse* response) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<RegisterPrometheusResponse> response_guard(response);

  if (cntl->Failed()) {
    XLOG(WARNING) << "Fail to send RegisterPrometheus, " << cntl->ErrorText();
    return;
  }
  XLOG(INFO) << "Success register prometheus from " << cntl->remote_side()
             << " name is " << response->name();
}

void ClientStub::HandlePushPrometheusResponse(
    brpc::Controller* cntl, google::protobuf::Empty* response) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<google::protobuf::Empty> response_guard(response);

  if (cntl->Failed()) {
    XLOG(WARNING) << "Fail to push prometheus "
                  << " error " << cntl->ErrorText();
    return;
  }
}

void ClientStub::HandleDeregisterPrometheusResponse(
    brpc::Controller* cntl, RegisterPrometheusResponse* response) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<RegisterPrometheusResponse> response_guard(response);
  if (cntl->Failed()) {
    XLOG(WARNING) << "Fail to deregister prometheus " << cntl->ErrorText();
    return;
  }
}

void ClientStub::HandlePushSignalFrameInfo(brpc::Controller* cntl,
                                           google::protobuf::Empty* response,
                                           int signal, int rank) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<google::protobuf::Empty> response_guard(response);
  if (cntl->Failed()) {
    XLOG(WARNING) << "Fail to push signal for rank " << rank << " signal "
                  << signal << " " << cntl->ErrorText();
    return;
  }
}

void ClientStub::requestRegisterPrometheus(
    bool sync, const std::string& kernel_name, const std::string& gauge_prefix,
    int rank, int local_rank,
    const std::map<std::string, std::string>& labels) {
  RegisterPrometheusRequest request;
  for (auto const& pair : labels) {
    (*request.mutable_labels())[pair.first] = pair.second;
  }
  request.set_gauge_prefix(gauge_prefix);
  request.set_rank(rank);
  request.set_local_rank(local_rank);
  request.set_kernel_name(kernel_name);
  RegisterPrometheusResponse* response = new RegisterPrometheusResponse();
  response->set_ret_code(1);
  brpc::Controller* cntl = new brpc::Controller();
  if (sync) {
    stub_->RegisterPrometheus(cntl, &request, response, NULL);
    ClientStub::HandleRegisterPrometheusResponse(cntl, response);
  } else {
    google::protobuf::Closure* done = brpc::NewCallback(
        &ClientStub::HandleRegisterPrometheusResponse, cntl, response);
    stub_->RegisterPrometheus(cntl, &request, response, done);
  }
}

void ClientStub::pushPrometheusMetrics(bool sync, const BrpcMetrics* request) {
  google::protobuf::Empty* response = new google::protobuf::Empty();
  brpc::Controller* cntl = new brpc::Controller();
  if (sync) {
    stub_->PushPrometheus(cntl, request, response, NULL);
    ClientStub::HandlePushPrometheusResponse(cntl, response);
  } else {
    google::protobuf::Closure* done = brpc::NewCallback(
        &ClientStub::HandlePushPrometheusResponse, cntl, response);
    stub_->PushPrometheus(cntl, request, response, done);
  }
}

void ClientStub::pushPrometheusMemMetrics(bool sync, const std::string& name,
                                          int local_rank, uint64_t counter) {
  BrpcMetrics request;
  MemBrpcMetrics* metrics = request.mutable_mem_metrics();

  metrics->set_counter(counter);
  request.set_name(name);
  request.set_local_rank(local_rank);

  pushPrometheusMetrics(sync, &request);
}

void ClientStub::pushPrometheusThroughputMetrics(
    bool sync, const std::string& name, int local_rank, uint64_t avg_latency,
    uint64_t min_latency, uint64_t max_latency, uint64_t p99_latency,
    double performance) {
  BrpcMetrics request;
  KernelBrpcMetrics* metrics = request.mutable_kernel_metrics();

  metrics->set_avg_latency(avg_latency);
  metrics->set_max_latency(max_latency);
  metrics->set_p99_latency(p99_latency);
  metrics->set_performance(performance);
  metrics->set_min_latency(min_latency);

  request.set_name(name);
  request.set_local_rank(local_rank);

  pushPrometheusMetrics(sync, &request);
}

void ClientStub::pushPrometheusCommonMetrics(
    bool sync, const std::string& name, int64_t local_rank, int64_t hang,
    int64_t start_dump, int64_t end_dump, int64_t pool_queue_size,
    int64_t work_queue_size, int64_t gc_count, int64_t data_loader_count) {
  BrpcMetrics request;
  CommonBrpcMetrics* common_metrics = request.mutable_common_metrics();

  common_metrics->add_metrics(hang);
  common_metrics->add_metrics(start_dump);
  common_metrics->add_metrics(end_dump);
  common_metrics->add_metrics(pool_queue_size);
  common_metrics->add_metrics(work_queue_size);
  common_metrics->add_metrics(gc_count);
  common_metrics->add_metrics(data_loader_count);

  request.set_name(name);
  request.set_local_rank(local_rank);

  pushPrometheusMetrics(sync, &request);
}

void ClientStub::requestDeRegisterPrometheus(bool sync, const std::string& name,
                                             int rank, int local_rank) {
  DeRegisterPrometheusRequest request;
  request.set_rank(rank);
  request.set_local_rank(local_rank);
  request.set_name(name);
  RegisterPrometheusResponse* response = new RegisterPrometheusResponse();
  response->set_ret_code(1);
  brpc::Controller* cntl = new brpc::Controller();
  if (sync) {
    stub_->DeRegisterPrometheus(cntl, &request, response, NULL);
    ClientStub::HandleDeregisterPrometheusResponse(cntl, response);
  } else {
    google::protobuf::Closure* done = brpc::NewCallback(
        &ClientStub::HandleDeregisterPrometheusResponse, cntl, response);
    stub_->DeRegisterPrometheus(cntl, &request, response, done);
  }
}

void ClientStub::pushSignalFrameInfo(const SignalFrameRequest* request) {
  // when program has been signaled, it will exit, the training process do
  // not care the response, so we async doing the response and sync call
  // on xpu_deamon_server
  google::protobuf::Empty* response = new google::protobuf::Empty();
  brpc::Controller* cntl = new brpc::Controller();
  google::protobuf::Closure* done =
      brpc::NewCallback(&ClientStub::HandlePushSignalFrameInfo, cntl, response,
                        request->signal(), request->rank());
  stub_->PushSignalFrameInfo(cntl, request, response, done);
}

}  // namespace server
}  // namespace xpu_timer
