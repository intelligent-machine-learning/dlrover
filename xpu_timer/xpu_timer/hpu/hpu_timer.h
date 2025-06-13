#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/platform.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/hpu/hpu_dtype_util.h"
#include "xpu_timer/protos/hook.pb.h"
namespace xpu_timer {
namespace hpu {
class InterceptManager;
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
  EventStartTimeHelper(aclrtStream s);
  void reset();
  time_t getTime(aclrtEvent kernel_launch_start, bool* is_validate_to_trace);

 private:
  // start event on this stream
  aclrtEvent start_event_;
  aclrtEvent stop_event_;
  // time in us
  time_t cpu_time_;
  aclrtStream stream_;
};

class HpuTimer : public XpuTimer {
  /* Use cuda event to timing kernel. */
 public:
  using InnerInterceptManager = InterceptManager;
  using FnReturn = std::tuple<const std::string, uint64_t, Labels>;

  explicit HpuTimer() {
    auto status = aclrtCreateEvent(&start_event_);
    if (status != ACL_ERROR_NONE) {
      XLOG(ERROR) << "record event err: " << aclGetRecentErrMsg()
                  << " stream: " << stream_;
    }
    status = aclrtCreateEvent(&stop_event_);
    if (status != ACL_ERROR_NONE) {
      XLOG(ERROR) << "record event err: " << aclGetRecentErrMsg()
                  << " stream: " << stream_;
    }
    hang_counter_ = 0;
    trace = new hook::KernelTrace();  // HpuTimer is pooling, trace should
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
  // dump the trace meta, mapping from trace_code -> kernel_name
  static void dumpTraceMeta(const std::string& path,
                            const std::vector<std::string>& extra);
  static void
  doPrepareForDumpTrace();  // reset start timer on each stream, it use to get
                            // timestamp for kernel when is running on GPU.

  /*
   * ===================================
   * public methods and vars
   * ===================================
   */

  void reset(aclrtStream s, std::function<HpuTimer::FnReturn(HpuTimer*)> cb,
             const std::string_view& type);

  // kernel trace object
  hook::KernelTrace* trace;

  // seqnum for each hccl comm, key is comm hash, value is auto inc from 0
  static std::unordered_map<uint64_t, uint64_t> hccl_seq_num;

  static double mmPerformance(uint64_t latency_in_us, uint64_t problem_size) {
    return (double)(problem_size) / latency_in_us / 1e6;  // Tflops
  };

  static double collPerformance(uint64_t latency_in_us, uint64_t problem_size) {
    return (double)(problem_size) / 1e3 / latency_in_us;  // Gbps
  }

  static std::string collBucketFn(double performance, uint64_t problem_size,
                                  Labels* label);

  static int getMatmulBucket(double performance, const std::string& dtype) {
    return (int)(performance / HpuDataTypeUtils::getGpuHardwareFlops(dtype));
  }

  static std::string matmulBucketFn(double peformance, uint64_t problem_size,
                                    Labels* label);

 private:
  void reset_cb(std::function<HpuTimer::FnReturn(HpuTimer*)> cb);
  void reset_stream(aclrtStream s, const std::string_view& type);

  aclrtEvent start_event_, stop_event_;  // owned
  aclrtStream stream_;                   // not owned
  // return kernel name, it's callback function and called in background thread,
  // be careful of the lifetime of object in closure.
  std::function<const std::string()> rebuild_cb_;
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
  static std::unordered_map<aclrtStream, EventStartTimeHelper*>
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

class InterceptManager {
 private:
  struct MatmulInfo {
    void* executor_addr_;
    std::array<int64_t, 4> bmnk_;
    aclDataType dtype_;
    std::string api_;
    std::string name_prefix_;

   public:
    MatmulInfo(void* executor_addr_, std::array<int64_t, 4> bmnk_,
               aclDataType dtype_, std::string api_, std::string name_prefix_)
        : executor_addr_(executor_addr_),
          bmnk_(bmnk_),
          dtype_(dtype_),
          api_(api_),
          name_prefix_(name_prefix_) {}
  };
  struct GroupedMamtulInfo {
    void* executor_addr_;
    std::vector<std::array<int64_t, 4>> bmnks_;
    aclDataType dtype_;
    std::string api_;
    std::string name_prefix_;

   public:
    GroupedMamtulInfo(void* executor_addr_,
                      std::vector<std::array<int64_t, 4>> bmnks_,
                      aclDataType dtype_, std::string api_,
                      std::string name_prefix_)
        : executor_addr_(executor_addr_),
          bmnks_(bmnks_),
          dtype_(dtype_),
          api_(api_),
          name_prefix_(name_prefix_) {}
  };
  std::unordered_map<void*, std::shared_ptr<InterceptManager::MatmulInfo>>
      matmul_info_map_;
  std::unordered_map<void*,
                     std::shared_ptr<InterceptManager::GroupedMamtulInfo>>
      grouped_matmul_info_map_;

 public:
  // matmul info
  void interceptMatmulInfo(const aclTensor* self, const aclTensor* other,
                           aclOpExecutor** executor);
  std::function<HpuTimer::FnReturn(HpuTimer*)> handleMatmul(
      aclOpExecutor* executor);
  std::function<HpuTimer::FnReturn(HpuTimer*)> handleHccl(
      uint64_t count, HcclDataType datatype, HcclComm& comm,
      const std::string& func_name, const std::string& coll_type);

  // grouped matmul info
  void interceptGroupedMatmulV2Info(const aclTensorList* x,
                                    const aclTensorList* weight,
                                    const aclIntArray* groupListOptional,
                                    int64_t splitItem, int64_t groupType,
                                    aclOpExecutor** executor);

  std::function<HpuTimer::FnReturn(HpuTimer*)> handleGroupedMatmulV2(
      aclOpExecutor* executor);
};
}  // namespace hpu
}  // namespace xpu_timer
