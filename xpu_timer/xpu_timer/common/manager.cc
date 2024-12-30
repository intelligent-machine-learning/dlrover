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

#include "xpu_timer/common/manager.h"

#include <signal.h>
#include <unistd.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process/extend.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/signal_handler.h"
#include "xpu_timer/common/version.h"
#include "xpu_timer/protos/hosting_service.pb.h"

#ifdef XPU_NVIDIA
#include "xpu_timer/nvidia/nvidia_timer.h"
using XPU_TIMER = xpu_timer::nvidia::NvidiaGpuTimer;
#endif

namespace bip = boost::interprocess;

namespace xpu_timer {

/*
 * ===================================
 * KernelTraceManager public functions
 * ===================================
 */
KernelTraceManager& KernelTraceManager::getInstance() {
  std::call_once(init_flag_, &KernelTraceManager::initSingleton);
  return *instance_;
}

void KernelTraceManager::initSingleton() {
  instance_ = new KernelTraceManager;
  instance_->kernel_trace_count_ =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_TIMELINE_TRACE_COUNT");

  instance_->kernel_trace_.set_pid(boost::this_process::get_id());
  instance_->kernel_trace_.set_rank(config::GlobalConfig::rank);

  instance_->switch_ = std::make_unique<util::ShmSwitch>(
      config::GlobalConfig::local_world_size + 1,
      config::GlobalConfig::local_rank, false);

  instance_->py_tracing_library_ =
      new py_tracing_manager::PyTracingLibrary("libpy_tracing.so");
  instance_->host_tracing_functions_.push_back("GC");
  instance_->host_tracing_functions_.push_back(
      "torch.utils.data.dataloader@_BaseDataLoaderIter@__next__");
  instance_->host_tracing_functions_.push_back("torch@cuda@synchronize");
  instance_->host_tracing_functions_.push_back("torch.cuda@Event@synchronize");
  instance_->host_tracing_functions_.push_back("torch.cuda@Event@wait");
  instance_->host_tracing_functions_.push_back("torch.cuda@Stream@synchronize");
  instance_->host_tracing_functions_.push_back("torch.cuda@Stream@wait_event");
  instance_->host_tracing_functions_.push_back("torch.cuda@Stream@wait_stream");
  instance_->host_tracing_functions_.push_back("torch@autograd@backward");
  instance_->host_tracing_functions_.push_back("torch@autograd@grad");
  instance_->host_tracing_functions_.push_back(
      "megatron.core.pipeline_parallel@schedules@forward_step");
  instance_->host_tracing_functions_.push_back(
      "megatron.core.pipeline_parallel@schedules@backward_step");
  std::string tracing_functions =
      EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_HOST_TRACING_FUNC");
  if (tracing_functions != util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
    std::vector<std::string> funcs = util::split(tracing_functions, ",");
    for (const auto& func : funcs)
      instance_->host_tracing_functions_.push_back(func);
  }

  std::vector<std::string> errors = instance_->py_tracing_library_->Register(
      instance_->host_tracing_functions_);
  for (size_t i = 0; i < instance_->host_tracing_functions_.size(); i++) {
    XLOG(INFO) << "Resiter host function "
               << instance_->host_tracing_functions_[i] << ",status "
               << errors[i];
  }
  instance_->reset("init");
  std::atexit([] { delete instance_; });
}

bool KernelTraceManager::triggerTrace() {
  if (switch_->getObj()->reset_flag) {
    // ensure all ranks into reseting and set reset_flag to false
    util::InterProcessBarrier(config::GlobalConfig::local_world_size,
                              config::GlobalConfig::local_rank,
                              "reset_trace_barrier");
    switch_->getObj()->reset_flag = false;
    XLOG(INFO) << "Reset all status";
    reset("reset");
    return false;
  }
  if (has_trigger_trace_) return true;
  if (switch_->getObj()->start_dump == 0) return false;
  auto now = std::chrono::system_clock::now();
  time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  int64_t now_timestamp = static_cast<std::int64_t>(now_time_t);
  if (now_timestamp >= switch_->getObj()->timestamp) {
    has_trigger_trace_ = true;
    py_tracing_library_->SwitchTracing(1);
    XLOG(INFO) << "Trigger trace after " << switch_->getObj()->timestamp;
    return true;
  }
  return false;
}

void KernelTraceManager::reset(const std::string& barrier_name) {
  has_do_prepare_for_dump_ = false;
  has_trigger_trace_ = false;
  py_tracing_library_->SwitchTracing(0);
  // make sure all training ranks into reset
  util::InterProcessBarrier(config::GlobalConfig::local_world_size,
                            config::GlobalConfig::local_rank, barrier_name);
  switch_->getObj()->start_dump = 0;
  XLOG(INFO) << barrier_name << " reset start_dump to "
             << switch_->getObj()->start_dump;
}

template <>
bool KernelTraceManager::pushTrace(XPU_TIMER* work_item);

bool KernelTraceManager::prepareDump() {
  // this function is not reentrant, we use has_do_prepare_for_dump_ to guard
  // that
  if (!has_do_prepare_for_dump_) {
    // has do prepare
    has_do_prepare_for_dump_ = true;
    // kernel_trace_count is changed on the fly, clean trace message
    kernel_dump_type_ = std::bitset<32>(switch_->getObj()->dump_kernel_type);
    XLOG(INFO) << "dump_kernel_type " << switch_->getObj()->dump_kernel_type;
    if (switch_->getObj()->dump_count != kernel_trace_count_) {
      kernel_trace_count_ = switch_->getObj()->dump_count;
    }
    XPU_TIMER::doPrepareForDumpTrace();
    return true;
  }
  // not prepare
  return false;
}

void KernelTraceManager::dumpPyTracing() {
  // fetch all host tracing data
  // 1. get current trace data
  // 2. get trace data from ready queue
  // 3. dump to protobuf
  // 4. return to pool of trace data
  for (size_t name_index = 0; name_index < host_tracing_functions_.size();
       name_index++) {
    std::vector<XpuTimerPyTracingDataArray*> holders;
    const std::string& name = host_tracing_functions_[name_index];
    XpuTimerPyTracingDataArray* tracing_data =
        py_tracing_library_->GetPartialTracingData(name_index);
    if (tracing_data) holders.push_back(tracing_data);
    while (1) {
      XpuTimerPyTracingDataArray* tracing_data =
          py_tracing_library_->GetFullTracingData(name_index);
      if (!tracing_data) break;
      holders.push_back(tracing_data);
    }
    if (!holders.size()) continue;
    int inner_index = 0;
    for (auto each_tracing_data : holders) {
      for (uint32_t i = 0; i < each_tracing_data->cur; i++) {
        if (each_tracing_data->data[i].start > last_kernel_launch_time_ ||
            each_tracing_data->data[i].start == 0)
          continue;
        auto trace = kernel_trace_.add_host_traces();
        trace->set_start_us(each_tracing_data->data[i].start);
        trace->set_dur_us(each_tracing_data->data[i].end -
                          each_tracing_data->data[i].start);
        trace->set_count(each_tracing_data->data[i].count);
        trace->set_name(name);
        // add gc debug data
        if (each_tracing_data->data[i].type == PAYLOAD_GC) {
          hook::GcDebugData* gc_debug = trace->mutable_gc_debug();
          gc_debug->set_collected(
              each_tracing_data->data[i].payload.gc_debug[0]);
          gc_debug->set_uncollectable(
              each_tracing_data->data[i].payload.gc_debug[1]);
        }
      }
      inner_index++;
    }
    for (auto each_tracing_data : holders)
      py_tracing_library_->ReturnTracingData(each_tracing_data,
                                             PY_TRACING_EMPTY_POOL, name_index);
  }
}

int64_t KernelTraceManager::getGcCount() {
  return py_tracing_library_->GetTracingCount(PY_TRACING_GC);
}

int64_t KernelTraceManager::getDataLoaderCount() {
  return py_tracing_library_->GetTracingCount(PY_TORCH_DATALOADER);
}

void uploadOss(const std::string& oss_bin_path,
               const server::DumpKernelTraceRequest::OssArgs oss_args,
               const std::string& timeline_path) {
  bp::environment env = boost::this_process::environment();
  env.erase("LD_PRELOAD");
  std::string oss_ak;
  std::string oss_sk;
  if (!util::AES128_CBC(oss_args.oss_ak(), &oss_ak)) {
    XLOG(INFO) << "AES128_CBC error, skip upload oss, ak:" << oss_args.oss_ak();
    return;
  }
  if (!util::AES128_CBC(oss_args.oss_sk(), &oss_sk)) {
    XLOG(INFO) << "AES128_CBC error, skip upload oss, sk:" << oss_args.oss_sk();
    return;
  }

  bp::ipstream out_stream;
  bp::ipstream err_stream;

  // ossutil -i ak -k sk -e endpoint cp -f -r files oss:// --include
  // 00000-00002*
  bp::child oss_cmd(oss_bin_path,
                    bp::args({
                        "-i",
                        oss_ak,
                        "-k",
                        oss_sk,
                        "-e",
                        oss_args.oss_endpoint(),
                        "cp",
                        "-f",
                        "-r",
                        timeline_path,
                        oss_args.oss_path(),
                        "--include",
                        util::getUniqueFileNameByCluster("*"),
                    }),
                    env, bp::std_out > out_stream, bp::std_err > err_stream);

  std::thread reader_stdout([&out_stream] {
    std::string line;
    while (std::getline(out_stream, line)) XLOG(INFO) << line;
  });

  std::thread reader_stderr([&err_stream] {
    std::string line;
    while (std::getline(err_stream, line)) XLOG(INFO) << line;
  });

  oss_cmd.wait();
  out_stream.pipe().close();
  err_stream.pipe().close();
  reader_stderr.join();
  reader_stdout.join();
  int exit_code = oss_cmd.exit_code();
  if (exit_code) {
    XLOG(WARNING) << "Call " << oss_bin_path << " error, code is " << exit_code;
    return;
  }
  XLOG(INFO) << "Upload to oss " << oss_args.oss_path();
}

void KernelTraceManager::dumpKernelTraceAndHostTrace(
    stack_util::PyStackInProcess* py_stack_util) {
  py_tracing_library_->SwitchTracing(0);
  const std::string& dump_path = switch_->getObj()->dump_path;
  util::ScopeGuard guard([this]() {
    reset("after_dump");
    kernel_trace_.mutable_traces()->Clear();
    kernel_trace_.mutable_host_traces()->Clear();
  });

  if (util::ensureDirExists(dump_path)) {
    XLOG(ERROR) << "Could not create dir for timeline.meta";
    return;
  }
  dumpPyTracing();
  // if 3d parallel, need to check the which rank to dump meta in each process
  // group.
  if (util::EnvVarRegistry::GetEnvVar<bool>("XPU_TIMER_ALL_DUMP_TIMELINE") ||
      config::GlobalConfig::rank == 0) {
    XPU_TIMER::dumpTraceMeta(dump_path, host_tracing_functions_);
    py_stack_util->dumpPyStack(dump_path, config::GlobalConfig::rank);
  }

  std::ostringstream oss;
  oss << dump_path << "/" << util::getUniqueFileNameByCluster(".timeline");
  std::string file_path(oss.str());
  std::ofstream file(file_path, std::ios::binary | std::ios::out);
  if (!file) {
    XLOG(FATAL) << "Error opening file for writing";
    return;
  }

  std::string binary_message;
  kernel_trace_.SerializeToString(&binary_message);
  file << binary_message;
  XLOG(INFO) << "Rank " << config::GlobalConfig::rank << " dump timeline to "
             << file_path << " with " << kernel_trace_count_ << " events";

  auto bin_path = bp::search_path("ossutil");
  const std::string& oss_bin_path = bin_path.string();
  server::DumpKernelTraceRequest::OssArgs oss_args;
  oss_args.ParseFromString(switch_->getObj()->oss_dump_args);
  if (oss_bin_path.empty() || oss_args.oss_path().empty()) {
    XLOG(INFO) << "No ossutil found or empty oss_path, skip upload";
    return;
  }
  std::thread uploading(uploadOss, oss_bin_path, oss_args, dump_path);
  uploading.detach();
}

template <>
bool KernelTraceManager::pushTrace(XPU_TIMER* work_item) {
  // return true means not ready to dump
  if (prepareDump()) return true;
  if (!kernel_dump_type_[work_item->trace->kernel_type()]) {
    return true;
  }
  if (static_cast<uint32_t>(kernel_trace_.traces_size()) ==
      kernel_trace_count_) {
    last_kernel_launch_time_ =
        kernel_trace_.traces(kernel_trace_count_ - 1).start_us();
    return false;
  }
  auto trace = work_item->trace;

  trace->set_trace_code(work_item->getTraceCode());
  trace->set_dur_us(work_item->getDuration());
  trace->set_trace_id(work_item->getTraceId());

  if (work_item->isHost()) {
    trace->set_start_us(work_item->getLaunchTimeStamp());
    trace->set_delay_us(0);
    trace->set_is_host(true);
    *kernel_trace_.add_traces() = *trace;
  } else {
    trace->set_start_us(work_item->getExecuteTimeStamp());
    int64_t delay_us = (int64_t)(work_item->trace->start_us()) -
                       work_item->getLaunchTimeStamp();
    delay_us = delay_us < 0 ? 0 : delay_us;
    trace->set_delay_us(uint32_t(delay_us));
    if (work_item->isValidateToTrace()) {
      *kernel_trace_.add_traces() = *trace;
    }
  }

  return true;
}

/*
 * ===================================
 * GpuTimerManager public functions
 * ===================================
 */
template <>
void GpuTimerManager<XPU_TIMER>::initSingleton();
template <>
void GpuTimerManager<XPU_TIMER>::stopWork() noexcept;
template <>
void GpuTimerManager<XPU_TIMER>::doWork();
template <>
GpuTimerManager<XPU_TIMER>& GpuTimerManager<XPU_TIMER>::getInstance();
template <>
void GpuTimerManager<XPU_TIMER>::startWork();
template <>
XPU_TIMER* GpuTimerManager<XPU_TIMER>::getEvent();
template <>
void GpuTimerManager<XPU_TIMER>::recordEvent(XPU_TIMER* event);
template <>
void GpuTimerManager<XPU_TIMER>::startDaemon(int port);
template <>
void GpuTimerManager<XPU_TIMER>::doHang();
template <>
void GpuTimerManager<XPU_TIMER>::pushItemsToMetricsManager(XPU_TIMER* event);
template <>
void GpuTimerManager<XPU_TIMER>::doHang() {
  if (has_do_hang_) {
    XLOG(INFO) << "Has dump stack, ignore";
    return;
  }
  has_do_hang_ = true;

  xpu_timer::metrics::CommonMetrics comm_metrics;
  comm_metrics.hang = 1;
  comm_metrics.start_dump = 1;
  auto start = std::chrono::system_clock::now();
  metrics_manager_->pushCommonMetricsToRemote("common", comm_metrics);
  XLOG(FATAL) << "Operator is hang, log stack";
  std::vector<std::string> hang_items;
  working_queue_.printHangName(&hang_items);
  std::string dump_path =
      ::xpu_timer::util::EnvVarRegistry::GetEnvVar<std::string>(
          "XPU_TIMER_TIMELINE_PATH");
  // {"--pid", "--rank", "--world_size", "--dump_path"}
  dump_stub_->requestDump(
      true, boost::this_process::get_id(), config::GlobalConfig::rank,
      config::GlobalConfig::world_size, dump_path, hang_items);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

  comm_metrics.end_dump = duration.count();
  metrics_manager_->pushCommonMetricsToRemote("common", comm_metrics);
  bool exit =
      ::xpu_timer::util::EnvVarRegistry::GetEnvVar<bool>("XPU_TIMER_HANG_KILL");
  if (exit) std::exit(0);
}
template <>
XPU_TIMER* GpuTimerManager<XPU_TIMER>::getEvent() {
  return event_pool_.getObject();
}

template <>
void GpuTimerManager<XPU_TIMER>::startDaemon(int port) {
  bp::environment env = boost::this_process::environment();
  env.erase("LD_PRELOAD");
  std::string xpu_daemon_path =
      EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_DAEMON_PATH");

  std::filesystem::path file_path{"/tmp/xpu_timer_daemon.pid"};
  if (std::filesystem::exists(file_path)) {
    std::ifstream pid_file(file_path);
    std::string pid_str;
    std::getline(pid_file, pid_str);
    bp::child::child_handle pid{std::stoi(pid_str)};
    std::error_code ec;
    bp::detail::api::terminate(pid, ec);
    XLOG(INFO) << "Find xpu timer daemon, kill it, code is " << ec;
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  auto env_setup = [](auto& exec) {
    // set session id, this for when local rank 0 exit, this process will find
    // init process to host it's self.
    setsid();
    // clang-format off
    // process topology, arrow -> means "parent->child"
    // main process -> daemon server ---> gdb dump process
    //                                `-> py-spy dump process
    //
    // We close all file descriptors, including stdin, stdout, and stderr, to prevent
    // the main process from hanging when reading from these streams. Since the
    // daemon server will not use standard output/error and continues running,
    // leaving these descriptors open could cause blocking I/O operations in the
    // parent, such as when using the `tee` command. Additionally, closing these
    // descriptors before calling execve ensures they are not improperly inherited
    // by the child(gdb,py-spy) process, which could otherwise lead to a loss of standard
    // output/error functionality in the child, by the way, child(gdb/py-spy) process use
    // anonymous pipe to communicate with daemon server in boost.
    // clang-format on
    for (int fd = 0; fd < sysconf(_SC_OPEN_MAX); ++fd) close(fd);
  };
  if (xpu_daemon_path == util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
    auto bin_path = bp::search_path("xpu_timer_daemon");
    xpu_daemon_path = bin_path.string();
  }

  std::string prometheus_port = std::to_string(port);

  XLOG(INFO) << "Xpu timer daemon path is " << xpu_daemon_path;
  instance_->daemon_ =
      new bp::child(xpu_daemon_path,
                    bp::args({
                        "--server-path",
                        daemon_addr_,
                        "--local_world_size",
                        std::to_string(config::GlobalConfig::local_world_size),
                        "--prometheus_port",
                        prometheus_port,
                        "--graceful_quit_on_sigterm",
                        "-usercode_in_pthread",
                        "-usercode_backup_threads",
                        "40",
                        "--reuse_uds_path",
                    }),
                    env, bp::extend::on_exec_setup = env_setup);
  instance_->daemon_->detach();
}

template <>
void GpuTimerManager<XPU_TIMER>::recordEvent(XPU_TIMER* event) {
  event->endRecord();
  // 10000 stack will captured
  if (!py_stack_util_->isFull()) {
    event->py_stack = new PyStack(std::move(py_stack_util_->getPyStack()));
  }
  working_queue_.push(event);
}

template <>
void GpuTimerManager<XPU_TIMER>::initSingleton() {
  // register env first
  util::REGISTER_ENV();
  instance_ = new GpuTimerManager<XPU_TIMER>;

  instance_->daemon_ = nullptr;
  instance_->dump_stub_ = nullptr;
  instance_->py_stack_util_ =
      new stack_util::PyStackInProcess("libpy_xpu_timer_callstack.so");
  instance_->startWork();

  std::atexit([] { delete instance_; });
}

template <>
void GpuTimerManager<XPU_TIMER>::stopWork() noexcept {
  if (!config::GlobalConfig::enable) return;

  should_run_.store(false);
  XLOG(INFO) << "Stoping poller thread...";
  if (event_poller_.joinable()) {
    event_poller_.join();
  }
  XLOG(INFO) << "Stoping Metrics manager";
  delete metrics_manager_;
  XLOG(INFO) << "Thread is stopped, exit process";
}

template <>
void GpuTimerManager<XPU_TIMER>::doWork() {
  bool need_sleep = true;
  time_t timeout = EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_HANG_TIMEOUT");
  bool is_ready = false;
  bool is_hang = false;
  auto is_hang_fn = [&is_hang, timeout](XPU_TIMER* value) -> bool {
    is_hang = value->isHang(timeout);
    return is_hang;
  };
  auto is_ready_fn = [&is_ready](XPU_TIMER* value) -> bool {
    is_ready = value->isReady();
    return is_ready;
  };
  int pool_queue_size = 0;
  int work_queue_size = 0;
  // main loop
  while (should_run_.load()) {
    if (need_sleep) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    XPU_TIMER* work_item =
        working_queue_.pop(is_hang_fn, is_ready_fn, &work_queue_size);
    if (!work_item) {
      need_sleep = true;
      continue;
    }
    if (is_hang) {
      doHang();
    }
    if (!is_ready) {
      need_sleep = true;
      continue;
    }

    // rebuild item, generate name
    work_item->reBuild();
    // generate kernel stack
    if (work_item->py_stack) {
      py_stack_util_->insertPyStack(work_item->getName(), *work_item->py_stack);
      delete work_item->py_stack;
      work_item->py_stack = nullptr;
    }

    if (loop_count_ >= 1000) {  // 1000 is for warmup

      // since 1000 event timeline about 5KiB, dump is fast, so we dump in
      // background thread blocking
      if (KernelTraceManager::getInstance().triggerTrace()) {
        // push return false means timeline buffer is full, and we can dump
        // timeline
        if (!KernelTraceManager::getInstance().pushTrace<XPU_TIMER>(
                work_item)) {
          KernelTraceManager::getInstance().dumpKernelTraceAndHostTrace(
              py_stack_util_);
        }
      }
      if (work_item->getProblemSize() == 0) {
        // if problem size is 0, return null, so we return object to pool
        event_pool_.returnObject(work_item, &pool_queue_size);
        continue;
      }
      pushItemsToMetricsManager(work_item);
    }
    event_pool_.returnObject(work_item, &pool_queue_size);
    if (loop_count_ % 50000 == 0) {  // about 10 seconds
      loop_count_ = 1001;            // warmup is 1000, sets to 1001.
      xpu_timer::metrics::CommonMetrics comm_metrics;
      comm_metrics.pool_queue_size = pool_queue_size;
      comm_metrics.work_queue_size = work_queue_size;
      comm_metrics.gc_count = KernelTraceManager::getInstance().getGcCount();
      comm_metrics.data_loader_count =
          KernelTraceManager::getInstance().getDataLoaderCount();

      if (!is_hang) {
        comm_metrics.hang = 0;
        comm_metrics.start_dump = 0;
        comm_metrics.end_dump = 0;
      }
      metrics_manager_->pushCommonMetricsToRemote("common", comm_metrics);
    }
    need_sleep = false;
    loop_count_++;
  }
}

template <>
GpuTimerManager<XPU_TIMER>& GpuTimerManager<XPU_TIMER>::getInstance() {
  std::call_once(init_flag_, &GpuTimerManager<XPU_TIMER>::initSingleton);
  return *instance_;
}

template <>
void GpuTimerManager<XPU_TIMER>::pushItemsToMetricsManager(
    XPU_TIMER* work_item) {
  if (work_item->getType() == constant::Metrics::MatmulMetrics::TYPE) {
    metrics_manager_->updateMetrics(work_item, XPU_TIMER::mmPerformance,
                                    XPU_TIMER::matmulBucketFn);
  } else if (work_item->getType() == constant::Metrics::CollMetrics::TYPE) {
    metrics_manager_->updateMetrics(work_item, XPU_TIMER::collPerformance,
                                    XPU_TIMER::collBucketFn);
  } else {
    metrics_manager_->updateMetrics(work_item, nullptr, nullptr);
  }
}

template <>
void GpuTimerManager<XPU_TIMER>::startWork() {
  // ===================================
  // Global config setup
  // ===================================
  config::setUpConfig();
  if (!config::GlobalConfig::enable) return;
  int local_rank = config::GlobalConfig::local_rank;
  int brpc_port;
  int prometheus_port = EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_PORT");
  if (prometheus_port != EnvVarRegistry::INT_DEFAULT_VALUE) {
    brpc_port = prometheus_port - 1;
  } else {
    int port_offset = util::config::getMinDeviceRank() *
                      20;  // num of devices is smaller than 20
    brpc_port =
        EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_BASEPORT") + port_offset;
    prometheus_port = brpc_port + 1;
  }
  std::string base_port = std::to_string(brpc_port);
  setLoggingPath(false);

  // ===================================
  // daemon server and client init
  // ===================================
  daemon_addr_ = "0.0.0.0:" + base_port;
  if (local_rank == 0) {
    // there has a barrier with all ranks in daemon thread.
    // see xpu_timer/server/server.cc
    startDaemon(prometheus_port);
    XLOG(INFO) << "start daemon at " << prometheus_port;
  }
  KernelTraceManager::getInstance();
  // barrier here wait daemon started, +1 for daemon server
  util::InterProcessBarrier(config::GlobalConfig::local_world_size + 1,
                            local_rank, "start_work_barrier");
  // if you want to use uds, check this
  // dump_stub_ = new ClientStub("unix:/tmp/xpu_timer_daemon.sock");
  dump_stub_ = std::make_shared<ClientStub>("127.0.0.1:" + base_port);
  metrics_manager_ = new MetricsManager(dump_stub_);

  // ===================================
  // static memeber setup
  // ===================================

  XPU_TIMER::doPrepare();
  SignalHandler::registerDefault(dump_stub_);
  should_run_.store(true);

  // ===================================
  // extern memeber setup
  // ===================================
  ::bvar::xpu_timer::bvar_enable_sampling_from_xpu_timer = true;
#ifdef _GNU_SOURCE
  // pthread_setname_np is not posix
  event_poller_ = std::thread(&GpuTimerManager::doWork, this);
  auto handle = event_poller_.native_handle();
  pthread_setname_np(handle, "xpu_poller");
#endif
}

}  // namespace xpu_timer
