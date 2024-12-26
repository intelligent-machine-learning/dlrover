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
#include <brpc/server.h>
#include <bvar/bvar.h>
#include <gflags/gflags.h>
#include <pthread.h>

#include <atomic>
#include <bitset>
#include <boost/process.hpp>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "xpu_timer/common/bvar_prometheus.h"
#include "xpu_timer/common/metrics.h"
#include "xpu_timer/common/stack_util.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/nvidia/nvidia_dtype_util.h"
#include "xpu_timer/protos/hook.pb.h"
#include "xpu_timer/python/py_tracing_data.h"
#include "xpu_timer/python/py_tracing_loader.h"
#include "xpu_timer/server/hosting_service_server_client.h"

namespace bp = ::boost::process;
namespace xpu_timer {
using namespace util;
using namespace server;
using namespace stack_util;
namespace gc_manager {
class PyGcLibrary;
}

class KernelTraceManager {
 public:
  // default trace events, not configurable. 1000 events of matmul and nccl is
  // enough.
  KernelTraceManager(const KernelTraceManager&) = delete;
  KernelTraceManager& operator=(const KernelTraceManager&) = delete;
  static KernelTraceManager& getInstance();

  // save trace data on disk, each trace file is 24Bytes.
  // timeline_path
  //   ├──00000-00002.timeline
  //   ├──00001-00002.timeline
  //   └── timeline.meta     // only rank0 dump
  void dumpKernelTraceAndHostTrace(stack_util::PyStackInProcess* py_stack_util);
  void dumpPyTracing();
  bool prepareDump();
  int64_t getGcCount();
  int64_t getDataLoaderCount();
  template <typename T>
  bool pushTrace(T* work_item);
  bool triggerTrace();

 private:
  inline static KernelTraceManager* instance_ = nullptr;
  inline static std::once_flag init_flag_;

  // how many trace we collected, based on environment variable
  uint32_t kernel_trace_count_;
  // we pre allocate trace buffers, use curr_ to index current trace to modify.
  uint32_t curr_;
  // all traces for timeline
  hook::KernelTraces kernel_trace_;
  // which type kernel to dump
  std::bitset<32> kernel_dump_type_;

  // For each dump, we will perform some preparation actions. This process is
  // not reentrant, meaning it cannot safely be interrupted or run concurrently.
  // Therefore, we use a flag to indicate whether to perform the preparation
  // actions or not.
  bool has_do_prepare_for_dump_;
  bool has_trigger_trace_;
  std::unique_ptr<util::ShmSwitch> switch_;

  std::vector<std::string> host_tracing_functions_;
  py_tracing_manager::PyTracingLibrary* py_tracing_library_;
  uint64_t last_kernel_launch_time_;

  KernelTraceManager() = default;

  ~KernelTraceManager() {}

  static void initSingleton();
  void reset(const std::string& barrier_name);
};

template <typename T>
class GpuTimerManager {
  /*
   * TODO check the overhead of mutex.
   * since the critical section is tiny(only take fron deque, deque will not
   * resize in std's impl) but we do not know how many thread will fail into
   * critical section, if have lots of contention, maybe we have to use
   * coroutine for queue execution or use global tid map to manage each thread.

                            working flow

                                │
          ┌─────────────────┐   │
          │  pooling queue  ◄───┼──────────┐
          └────────┬────────┘   │          │
                   │1 get obj   │          │
          ┌────────▼────────┐   │          │
          │  get event      │   │          │7  return obj
          └────────┬────────┘   │  ┌───────┴────────┐
                   │2           │  │  compute time  │
          ┌────────▼────────┐   │  │  pushing data  │
          │  reset event    │   │  │  save timeline │
          │  record event   │   │  └───────▲────────┘
          └────────┬────────┘   │          │
                   │3           │          │
          ┌────────▼────────┐   │          │
          │ launch kernel   │   │          │
          └────────┬────────┘   │          │6
                   │4           │          │
          ┌────────▼────────┐  5│  ┌───────┴────────┐
          │  record event   ├───┼──►  working queue │
          └─────────────────┘   │  └────────────────┘
                                │
              main thread       │     background
                                       thread

   */
 public:
  GpuTimerManager(const GpuTimerManager&) = delete;
  GpuTimerManager& operator=(const GpuTimerManager&) = delete;

  static GpuTimerManager& getInstance();
  // get event from pool, event is inherit by XpuTimer
  T* getEvent();
  // record current, depends on platform
  void recordEvent(T* event);
  typename T::InnerInterceptManager intercept_manager;

 private:
  GpuTimerManager() = default;

  ~GpuTimerManager() { stopWork(); }

  inline static GpuTimerManager* instance_ = nullptr;
  inline static std::once_flag init_flag_;
  static void initSingleton();

  // event queue, running in infinity loop with 100us sleep.
  BlockingDeque<T> working_queue_;
  // pool of tracing event
  TimerPool<T> event_pool_;
  std::atomic<bool> should_run_{false};
  // working thread, exit when process id donw.
  std::thread event_poller_;
  // metrics manager own
  // get bvar and prometheus object depends on T.

  void pushItemsToMetricsManager(T* work_item);
  // dump stack /prometheus client
  std::shared_ptr<ClientStub> dump_stub_;
  // rank0 start a new process help to dump stack
  bp::child* daemon_;
  // daemon server addr, uds or ip
  std::string daemon_addr_;
  MetricsManager* metrics_manager_;

  // dump python stack in launch thread
  stack_util::PyStackInProcess* py_stack_util_;
  bool has_do_hang_{false};
  uint64_t loop_count_{1};

  // shutdown thread, called by atexit
  void stopWork() noexcept;
  // main loop
  void doWork();
  void startWork();
  void startDaemon(int port);
  void doHang();
  void deregisterMetrics();
  void pushremoteMetrics();
};

}  // namespace xpu_timer
