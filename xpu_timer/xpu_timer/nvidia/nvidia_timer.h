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

#include <dlfcn.h>

#include <array>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process.hpp>
#include <boost/process/extend.hpp>
#include <chrono>
#include <functional>
#include <initializer_list>
#include <map>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/platform.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/nvidia/nvidia_dtype_util.h"
#include "xpu_timer/protos/hook.pb.h"
namespace bp = ::boost::process;

namespace xpu_timer {
namespace nvidia {

class NcclCommWrapper {
 public:
#define DEF_COMM_INFO(FIELD, TYPE)                   \
  typedef TYPE (*get_Comm_##FIELD##_Fn)(ncclComm_t); \
  TYPE FIELD;

  DEF_COMM_INFO(commHash, uint64_t)
  DEF_COMM_INFO(rank, int)
  DEF_COMM_INFO(nRanks, int)
  DEF_COMM_INFO(nNodes, int)
  DEF_COMM_INFO(devComm, void*)
#undef DEF_COMM_INFO

 public:
  NcclCommWrapper(ncclComm_t comm);

#if defined(XPU_NVIDIA)
 public:
  static void* handle;  // dlopen libparse_params.so
  static std::string getNcclVersion();
  static void registerFunction();

#define DEF_COMM_INFO_FUNCTION(FIELD, TYPE) \
  static get_Comm_##FIELD##_Fn get_Comm_##FIELD##_func;

  DEF_COMM_INFO_FUNCTION(commHash, uint64_t)
  DEF_COMM_INFO_FUNCTION(rank, int)
  DEF_COMM_INFO_FUNCTION(nRanks, int)
  DEF_COMM_INFO_FUNCTION(nNodes, int)
  DEF_COMM_INFO_FUNCTION(devComm, void*)
#undef DEF_COMM_INFO_FUNCTION
#endif  // XPU_NVIDIA
};

namespace config = xpu_timer::util::config;
class EventStartTimeHelper {
  /* Using for getting when does the `torch kernel` real running. Record helper
   event on
   * this stream and synchronize it, and then get cpu time immediately, so we
   can use cpu time
   * to approximate time of helper event, and finally we can get time of `torch
   kernel`.
   *
   * kernel start time = cpu time + (kernel launch event - helper start event)
   *
                        cpu time
                            │
        start event         │         torch kernel
    synchronize time──────┐ │         elspesd time
                          │ │           ──────────
       reset helper ────┐ │ │          /          \
       start event      │ │ │         /            \
                        ▼ ▼ ▼        /              \
             ───────────────────────────────────────────────────► time
                       /           /▲               ▲
                      /___________/ │               │
                     /              │               │
                    /            kernel          kernel
     elspsed time  /          launch event     end event
     between helper event
     and torch kernel
     */

 public:
  EventStartTimeHelper(cudaStream_t s);
  void reset();
  time_t getTime(cudaEvent_t kernel_launch_start, bool* is_validate_to_trace);

 private:
  // start event on this stream
  cudaEvent_t start_event_;
  // time in us
  time_t cpu_time_;
  cudaStream_t stream_;
};

class InterceptManager;

class NvidiaGpuTimer : public XpuTimer {
  /* Use cuda event to timing kernel. */
 public:
  using FnReturn = std::tuple<const std::string, uint64_t, Labels>;
  using InnerInterceptManager = InterceptManager;

  explicit NvidiaGpuTimer() {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    hang_counter_ = 0;
    trace = new hook::KernelTrace();  // NvidiaGpuTimer is pooling, trace should
                                      // never free.
  }

  /*
   * ===================================
   * Interface Overrides
   * ===================================
   * overrides from the XpuTimer
   * ===================================
   */
  void startRecord() override;
  void endRecord() override;
  bool isReady() override;
  void reBuild() override;
  uint64_t getDuration() override;
  const std::string getName() override;
  const std::string_view& getType() override;
  const uint64_t getProblemSize() override;
  time_t getExecuteTimeStamp() override;
  time_t getLaunchTimeStamp() override;
  int getTraceCode() override;
  uint64_t getTraceId() override;
  Labels getExtraLabels() override;
  bool isHang(time_t timeout) override;
  bool ignoreHang() override;
  bool isHost() override;
  bool isValidateToTrace() override;

  /*
   * ===================================
   * Static overload
   * ===================================
   * Overload from the XpuTimer
   * ===================================
   */
  static void doPrepare();  // parse nccl syms if needed.
  static void dumpTraceMeta(const std::string& path,
                            const std::vector<std::string>&
                                extra);  // dump timeline tracing meta info
  static void
  doPrepareForDumpTrace();  // reset start timer on each stream, it use to get
                            // timestamp for kernel when is running on GPU.

  /*
   * ===================================
   * public methods and vars
   * ===================================
   */

  // the event is in object pool, we reset it by different kernel.
  // this is for nccl kernel
  void reset(cudaStream_t s,
             std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb,
             const std::string_view& type);
  // this is for matmul kernel
  void reset(cudaStream_t s, const std::string_view& type,
             const std::initializer_list<int64_t>& bmnk,
             const std::initializer_list<int64_t>& ld,
             const std::initializer_list<int64_t>& stride, const int trans_a,
             const int trans_b, const int algo, const std::string&& name_prefix,
             cudaDataType_t dtype, const std::string& api, uint8_t bias = 0);
  void reset(std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb,
             const std::string_view& type);
  // kernel trace object
  hook::KernelTrace* trace;

  // seqnum for each nccl comm, key is comm hash, value is auto inc from 0
  static std::unordered_map<uint64_t, uint64_t> nccl_seq_num;

  static double mmPerformance(uint64_t latency_in_us, uint64_t problem_size) {
    return (double)(problem_size) / latency_in_us / 1e6;  // Tflops
  };

  static double collPerformance(uint64_t latency_in_us, uint64_t problem_size) {
    return (double)(problem_size) / 1e3 / latency_in_us;  // Gbps
  }

  static std::string collBucketFn(double performance, uint64_t problem_size,
                                  Labels* label);

  static int getMatmulBucket(double performance, const std::string& dtype) {
    return (int)(performance / CudaDataTypeUtils::getGpuHardwareFlops(dtype));
  }

  static std::string matmulBucketFn(double peformance, uint64_t problem_size,
                                    Labels* label);

 private:
  void reset_cb(std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb);
  void reset_stream(cudaStream_t s, const std::string_view& type);

  class MatmulBuilderCallback {
   public:
    static constexpr int UNUSE = -1;
    MatmulBuilderCallback() {};
    NvidiaGpuTimer::FnReturn operator()(NvidiaGpuTimer*);
    void reset(const std::initializer_list<int64_t>& bmnk,
               const std::initializer_list<int64_t>& ld,
               const std::initializer_list<int64_t>& stride, const int trans_a,
               const int trans_b, const int algo,
               const std::string&& name_prefix, cudaDataType_t dtype,
               const std::string& api, uint8_t bias);

   private:
    std::array<int, 4> bmnk_;
    std::array<int, 3> ld_;
    std::array<int64_t, 3> stride_;
    int trans_a_;
    int trans_b_;
    int algo_;
    std::string name_prefix_;
    std::string api_;
    cudaDataType_t dtype_;
    uint8_t bias_;
  };

  cudaEvent_t start_event_, stop_event_;  // owned
  cudaStream_t stream_;                   // not owned
  // return kernel name, it's callback function and called in background thread,
  // be careful of the lifetime of object in closure.
  std::function<const std::string()> rebuild_cb_;
  // callable for matmul, compute flop
  MatmulBuilderCallback inner_rebuild_cb_;
  // kernel type, current is batched matmul, matmul, coll
  std::string_view type_;
  // kernel name with params concat together.
  std::string name_;
  // record when does kernel launch or host call
  time_t launch_time_timestamp_;
  std::chrono::time_point<std::chrono::system_clock> launch_time_;
  // counting down for detecting kernel hang
  std::chrono::time_point<std::chrono::system_clock> launch_time_for_hang_;

  // for matmul/fa, is tflop, for communication, is Gbits
  uint64_t problem_size_;
  // labels for prometheus.
  Labels extra_labels_;

  // id of this kernel with param, it's used as KernelTrace::trace_code
  int trace_code_;

  // auto incremented counter by name, it use to generate code in
  // KernelTrace::trace_code
  static std::unordered_map<std::string, int> tracing_metas_;
  // global counter for encoding kernel name with params to int.
  static int kernel_encoding_counter_;
  // record launch count for each kernel, it's useful for compare kernel
  // cross differnts rank, it will be arg in chrome trace json file.
  // key is from tracing_metas_, value is count.
  static std::unordered_map<int, uint64_t> trace_id_counter_;
  // trace id for each type of kernel
  uint64_t trace_id_;
  // each stream has a helper to get kernel real running time,
  static std::unordered_map<cudaStream_t, EventStartTimeHelper*>
      stream_timer_helper_;
  // hang counter, poller interval is 100us, hang counter added to
  // 10000, we check the timeout timestamp.
  uint64_t hang_counter_;
  constexpr static uint64_t hang_counter_estimator_ = 10000;
  // is torch.dist.barrier op
  bool is_barrier_;
  // is on host
  bool is_host_;
  std::chrono::time_point<std::chrono::system_clock> finish_time_;
  // Some events do not need to be traced, such as events that begin before the
  // EventStartTimeHelper.
  bool is_validate_to_trace_;
};

class FaParser : LibraryLoader {
 public:
  FaParser(const std::string& library_path);
  std::vector<uint64_t> getFaFwdShape(void**);
  std::vector<uint64_t> getFaBwdShape(void**);

 private:
  using getShapeFn = std::vector<uint64_t> (*)(void**);
  getShapeFn get_fwd_shape_;
  getShapeFn get_bwd_shape_;
  void LoadFn();
};

using InterceptSymbolPb = ::xpu_timer::hook::InterceptSymbol;

struct InterceptSymbol {
  explicit InterceptSymbol(const InterceptSymbolPb& pb);
  std::string func_name;
  std::string coll_type;
  std::string algo;
  std::string operation;
  std::string dtype;
  std::string func_type;
  bool only_trace;
};

class InterceptManager {
 public:
  enum SendRecvType {
    NoSendOrRecv = 0,  // no send or recv
    Send = 1,          // send
    Recv = 2           // recv
  };

 private:
  FaParser fa_parser_{"libparse_params.so"};
  ptrdiff_t getOffset(const void* symbol);
  std::vector<uint64_t> getFaFwdShape(void** args);
  std::vector<uint64_t> getFaBwdShape(void** args);
  std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> handleFa(
      void** args, const InterceptSymbol* sym);
  std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> handleNccl(
#if defined(CUDA_LAUNCH_EXC)
      const cudaLaunchConfig_t* config,
#else
      const void* config,  // not used
#endif
      const void* func, void** args, const InterceptSymbol* sym, bool* skip);

  struct NcclInfo {
    size_t count;
    std::shared_ptr<NcclCommWrapper> comm;
    ncclDataType_t datatype;
    SendRecvType send_recv_type;
  };

  // Record NCCL info from NCCL kernel like ncclAllReduce, ncclAllGather.
  // key: the address of devComm, which is the member of comm (ncclComm_t type).
  // value: the queue contains NCCL info with the same devCom.
  static std::unordered_map<void*, std::shared_ptr<std::queue<NcclInfo>>>
      nccl_info_map_;
  static std::unordered_set<const void*> fns_to_skip_;
  static std::unordered_map<const void*, const InterceptSymbol*> fns_to_name_;
  static std::unordered_map<ptrdiff_t, const InterceptSymbol> addr_to_name_;

  static std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
      cuda_launch_kernel_exc_default_;
  static std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
      cuda_launch_kernel_default_;
  static bool skip_tp_;
  static int tp_size_;

 public:
  bool isIntercepted(const void* func, const InterceptSymbol** sym);
#if defined(CUDA_LAUNCH_EXC)
  std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
  handleCudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func,
                            void** args, const InterceptSymbol* sym,
                            bool* skip);
#endif
  std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
  handleCudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                         void** args, size_t sharedMem, cudaStream_t stream,
                         const InterceptSymbol* sym, bool* skip);
  std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> deviceMemory(
      const std::string& name, const size_t size, const std::string& kind,
      bool is_host);

  template <bool skip = false>
  void interceptNcclInfo(size_t count, ncclDataType_t datatype,
                         ncclComm_t orig_comm, cudaStream_t stream,
                         SendRecvType send_recv_type = NoSendOrRecv) {
    // "cudaLaunchKernelExC()" will bot be executed when comm->nRanks <= 1.
    auto comm = std::make_shared<NcclCommWrapper>(orig_comm);
    if (comm->nRanks <= 1) {
      return;
    }
    if constexpr (skip) {
      // if in same node and coll is AllReduce/AllGather/ReduceScatter, is tp,
      // current we skip tp for better performance. see
      // https://forums.developer.nvidia.com/t/overhead-of-cudaeventrecord-cudalaunchkernelexc-in-multithreading/300769
      // in detail
      if (skip_tp_ && comm->nNodes == 1) return;
    }
    // In NVIDIA, the address of ncclComm_t->devComm is used as the key, whereas
    // in HPU, the value of ncllComm_t->devComm is used as the key. To
    // distinguish between them, we implement two different constructor
    // functions in the NcclCommWrapper.
    void* devcom_t = comm->devComm;
    auto it = nccl_info_map_.find(devcom_t);
    if (it == nccl_info_map_.end()) {
      auto q = std::make_shared<std::queue<NcclInfo>>();
      q->push({count, comm, datatype, send_recv_type});
      nccl_info_map_.emplace(devcom_t, q);
    } else {
      it->second->push({count, comm, datatype, send_recv_type});
    }
  }
  static void resetSymsMap();
  static void setUp();
};

}  // namespace nvidia
}  // namespace xpu_timer
