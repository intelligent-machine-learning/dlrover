#include "xpu_timer/nvidia/nvidia_timer.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/nvidia/cupti_activity_event.h"

namespace xpu_timer {
namespace nvidia {

/*
 * ===================================
 * Class impl of EventStartTimeHelper
 * ===================================
 */

EventStartTimeHelper::EventStartTimeHelper(cudaStream_t s) : stream_(s) {
  int rank = util::EnvVarRegistry::GetEnvVar<int>("LOCAL_RANK");
  cudaSetDevice(rank);
  auto status = cudaEventCreate(&start_event_);
  if (status != cudaSuccess) {
    XLOG(ERROR) << "create event err: " << cudaGetErrorString(status);
  }
}

void EventStartTimeHelper::reset() {
  auto status = cudaEventRecord(start_event_, stream_);
  if (status != cudaSuccess) {
    XLOG(ERROR) << "record event err: " << cudaGetErrorString(status)
                << " stream: " << stream_;
  }
  status = cudaEventSynchronize(start_event_);
  if (status != cudaSuccess) {
    XLOG(ERROR) << "sync event err: " << cudaGetErrorString(status)
                << " stream: " << stream_;
  }
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();
  cpu_time_ = std::chrono::duration_cast<std::chrono::microseconds>(
                  start.time_since_epoch())
                  .count();
}

time_t EventStartTimeHelper::getTime(cudaEvent_t start_launch_event,
                                     bool* is_validate_to_trace) {
  float elapsed_time;  // ms
  cudaEventElapsedTime(&elapsed_time, start_event_, start_launch_event);
  double es = ((double)elapsed_time * 1000);  // ms->us
  // Skip events that begin before the EventStartTimeHelper.
  *is_validate_to_trace = (es > 0);
  return cpu_time_ + time_t(es);
}

/*
 * ===================================
 * Class impl of NvidiaGpuTimer
 * ===================================
 * ===================================
 * Static variables
 * ===================================
 */

std::unordered_map<uint64_t, uint64_t> NvidiaGpuTimer::nccl_seq_num{};
std::unordered_map<std::string, int> NvidiaGpuTimer::tracing_metas_{};
std::unordered_map<int, uint64_t> NvidiaGpuTimer::trace_id_counter_{};
std::unordered_map<cudaStream_t, EventStartTimeHelper*>
    NvidiaGpuTimer::stream_timer_helper_{};
int NvidiaGpuTimer::kernel_encoding_counter_(0);
std::vector<xpu_timer::nvidia::NcclGroupCollInfo*>
    NvidiaGpuTimer::nccl_group_coll_queue;

thread_local uint32_t NvidiaGpuTimer::nccl_group_start_count = 0;

/*
 * ===================================
 * Interface Overrides
 * ===================================
 * The following member functions are
 * overrides from the base interface.
 * ===================================
 */
const std::string_view& NvidiaGpuTimer::getType() { return type_; }

const uint64_t NvidiaGpuTimer::getProblemSize() { return problem_size_; }

void NvidiaGpuTimer::startRecord() {
  is_in_nccl_group_ = false;
#if defined NVIDIA_WITH_CUPTI
  if (nccl_group_start_count) {
    is_in_nccl_group_ = true;
  }
  cupti_external_id_ =
      CuptiActivityEvent::getInstance()->pushCuptiExternalActivity();
#if 0
  XLOG(INFO) << "start record: " << cupti_external_id_;
#endif

#else
  cudaEventRecord(start_event_, stream_);
#endif
}

void NvidiaGpuTimer::endRecord() {
  if (is_host_) {
    finish_time_ = std::chrono::system_clock::now();
    return;
  }
#if defined NVIDIA_WITH_CUPTI
  CuptiActivityEvent::getInstance()->popCuptiExternalActivity();
#endif
  cudaEventRecord(stop_event_, stream_);
}

int NvidiaGpuTimer::getTraceCode() { return trace_code_; }

uint64_t NvidiaGpuTimer::getTraceId() { return trace_id_; }

const std::string NvidiaGpuTimer::getName() { return name_; }

bool NvidiaGpuTimer::isReady() {
  if (is_host_) return true;

#ifdef NVIDIA_WITH_CUPTI
  if (!CuptiActivityEvent::getInstance()->hasNextCompletedExternalId()) {
    if (cudaEventQuery(stop_event_) == cudaErrorNotReady) {
      hang_counter_ += 1;
    }
    return false;
  }

  while (CuptiActivityEvent::getInstance()->hasNextCompletedExternalId() &&
         (CuptiActivityEvent::getInstance()->getNextCompletedExternalId() ==
          cupti_external_id_)) {
    CuptiActivityEvent::getInstance()->popNextCompletedExternalId();
    uint64_t duration = nvidia_launch_info_.duration_;
    CuptiActivityEvent::getInstance()->getCuptiInfoByExternalId(
        cupti_external_id_, &nvidia_launch_info_);
    nvidia_launch_info_.duration_ += duration;
    may_ready = true;
  }

  if (may_ready &&
      CuptiActivityEvent::getInstance()->hasNextCompletedExternalId() &&
      (CuptiActivityEvent::getInstance()->getNextCompletedExternalId() !=
       cupti_external_id_)) {
#if 0
    XLOG(INFO) << "readyid: " << cupti_external_id_;
#endif

    return true;
  }

  if (cudaEventQuery(stop_event_) == cudaErrorNotReady) {
    hang_counter_ += 1;
  }

  return false;

#else
  bool ready = cudaEventQuery(stop_event_) != cudaErrorNotReady;

  if (!ready) {
    hang_counter_ += 1;
    return ready;
  }

  return ready;
#endif  // NVIDIA_WITH_CUPTI
}

time_t NvidiaGpuTimer::getExecuteTimeStamp() {
#ifndef NVIDIA_WITH_CUPTI
  nvidia_launch_info_.start_timestamp_ = stream_timer_helper_[stream_]->getTime(
      start_event_, &(this->is_validate_to_trace_));
#endif
  return nvidia_launch_info_.start_timestamp_;
}

time_t NvidiaGpuTimer::getLaunchTimeStamp() { return launch_time_timestamp_; }

uint64_t NvidiaGpuTimer::getDuration() {
  if (is_host_) {
    auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
        finish_time_ - launch_time_);
    return dur_us.count();
  }
#ifndef NVIDIA_WITH_CUPTI
  float elapsed_time;  // ms
  cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_);
  nvidia_launch_info_.duration_ = uint64_t(elapsed_time * 1000);
#endif
  return nvidia_launch_info_.duration_;
}

void NvidiaGpuTimer::reBuild() { name_ = rebuild_cb_(); }

Labels NvidiaGpuTimer::getExtraLabels() { return extra_labels_; }

bool NvidiaGpuTimer::isHang(time_t timeout) {
  if (hang_counter_ < hang_counter_estimator_) return false;
  hang_counter_ = 0;

  static const std::chrono::seconds timeout_second =
      std::chrono::seconds(timeout);
  if (std::chrono::system_clock::now() - launch_time_for_hang_ >
      timeout_second) {
    launch_time_for_hang_ = std::chrono::system_clock::now();
    return true;
  }
  return false;
}

bool NvidiaGpuTimer::isSkipped() {
  if (is_in_nccl_group_ &&
      (type_ == xpu_timer::constant::Metrics::CollMetrics::TYPE)) {
    reBuild();
#if 0
    XLOG(INFO) << "is Skipped in nccl group:  " << cupti_external_id_
               << " name: " << name_;
#endif
    return true;
  }

#ifdef NVIDIA_WITH_CUPTI
  if (!CuptiActivityEvent::getInstance()->hasNextCompletedExternalId()) {
    return false;
  }

  if (!may_ready &&
      (CuptiActivityEvent::getInstance()->getNextCompletedExternalId() !=
       cupti_external_id_) &&
      (cudaEventQuery(stop_event_) != cudaErrorNotReady)) {
    reBuild();
#if 0
    XLOG(INFO)
        << "is Skipped: event is prepared, but Cuptid is not hooked, "
           "nvidia_timer_id: "
        << cupti_external_id_ << " name: " << name_ << " cupti_queue_id: "
        << CuptiActivityEvent::getInstance()->getNextCompletedExternalId();
#endif
    return true;
  }
#endif

  return false;
}

bool NvidiaGpuTimer::ignoreHang() { return is_barrier_; }

bool NvidiaGpuTimer::isHost() { return is_host_; }

/*
 * ===================================
 * Static Overload
 * ===================================
 * The following member functions are
 * overload from the base interface.
 * ===================================
 */

void NvidiaGpuTimer::doPrepareForDumpTrace() {
  for (auto it : stream_timer_helper_) it.second->reset();
}

void NvidiaGpuTimer::doPrepare() {
  InterceptManager::setUp();

  std::string device_name_from_env =
      util::EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_DEVICE_NAME");

  if (device_name_from_env != util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
    XLOG(INFO) << "Device type set to " << device_name_from_env;
    CudaDataTypeUtils::setGpu(device_name_from_env);
  } else {
    CudaDataTypeUtils::setGpu(platform::getDeviceName());
  }
#ifdef NVIDIA_WITH_CUPTI
  CuptiActivityEvent::getInstance()->initCuptiActivity();
#endif
}

void NvidiaGpuTimer::doDestroy() {
#ifdef NVIDIA_WITH_CUPTI
  CuptiActivityEvent::getInstance()->closeCuptiActivity();
#endif
}

void NvidiaGpuTimer::dumpTraceMeta(const std::string& path,
                                   const std::vector<std::string>& extra) {
  if (util::ensureDirExists(path)) {
    XLOG(ERROR) << "Could not create dir for timeline.meta";
    return;
  }
  std::filesystem::path dir(path);
  std::filesystem::path file_path =
      dir / util::getUniqueFileNameByCluster(".timeline.meta");
  ;
  std::ostringstream oss;
  for (const auto& it : tracing_metas_) {
    oss << it.second << "," << it.first << std::endl;
  }
  for (const auto& it : extra) {
    oss << "xpu_timer_host_trace," << it << std::endl;
  }
  std::ofstream file(file_path);
  if (!file) {
    XLOG(ERROR) << "Could not create dir for timeline.meta";
    return;
  }
  file << oss.str();
}

std::string NvidiaGpuTimer::collBucketFn(double performance,
                                         uint64_t problem_size, Labels* label) {
  if (problem_size <= 8192) {  // 8192 bits
    (*label)["small"] = "1";
  } else {
    (*label)["small"] = "0";
  }
  double throughtput_gbps = performance;
  int level =
      int(throughtput_gbps * config::BvarMetricsConfig::comm_bucket_count /
          config::BvarMetricsConfig::nic_bandwidth_gbps);  // bucketing up to
                                                           // 400Gbps
  std::string string_level;
  std::string bucket_name;
  if (level > config::BvarMetricsConfig::comm_bucket_count) {
    level = config::BvarMetricsConfig::comm_bucket_count;
    string_level = std::to_string(config::BvarMetricsConfig::comm_bucket_count);
  } else {
    string_level = std::to_string(level + 1);
  }
  (*label)["level"] = string_level;
  static std::string coll_p =
      std::string(constant::Metrics::CollMetrics::BUCKET_NAME);
  bucket_name =
      coll_p + string_level + (*label)["operation"] + (*label)["algorithm"];
  return bucket_name;
};

std::string NvidiaGpuTimer::matmulBucketFn(double peformance,
                                           uint64_t problem_size,
                                           Labels* label) {
  double tflops = peformance;
  int level = getMatmulBucket(
      tflops * config::BvarMetricsConfig::mm_bucket_count, (*label)["dtype"]);
  std::string string_level;
  std::string bucket_name;
  if (level > config::BvarMetricsConfig::mm_bucket_count) {
    level = config::BvarMetricsConfig::mm_bucket_count;
    // tflops of flash_attn may higher than hardware tflops, add it to extra
    // level
    string_level =
        std::to_string(config::BvarMetricsConfig::mm_bucket_count + 1);
  } else {
    string_level = std::to_string(level + 1);
  }
  (*label)["level"] = string_level;

  static std::string compute_p =
      std::string(constant::Metrics::MatmulMetrics::BUCKET_NAME);
  bucket_name = compute_p + string_level + (*label)["operation"];
  return bucket_name;
};

/*
 * ===================================
 * Public methods
 * ===================================
 */

void NvidiaGpuTimer::reset_cb(
    std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb) {
  auto rebuild_fn = [this, cb]() {
    auto tup = cb(this);
    std::string name;
    std::tie(name, problem_size_, extra_labels_) = tup;

    if (tracing_metas_.find(name) == tracing_metas_.end()) {
      tracing_metas_[name] = kernel_encoding_counter_++;
    }

    trace_code_ = tracing_metas_[name];
    trace_id_ = ++trace_id_counter_[trace_code_];

    if (is_host_) {
      finish_time_ = std::chrono::system_clock::now();
    } else {
#ifndef NVIDIA_WITH_CUPTI
      auto it = stream_timer_helper_.find(stream_);
      if (it == stream_timer_helper_.end()) {
        stream_timer_helper_.emplace(stream_,
                                     new EventStartTimeHelper(stream_));
      }
#endif
    }

    launch_time_timestamp_ =
        std::chrono::duration_cast<std::chrono::microseconds>(
            launch_time_.time_since_epoch())
            .count();
    is_barrier_ = false;
    // is allreduce and reduce size is 8 bits, dtype is at::kByte, alias to
    // ncclUint8
    auto it_label = extra_labels_.find("operation");
    if (it_label != extra_labels_.end())
      is_barrier_ = false;
    else if (problem_size_ == 8 && it_label->second == "AllReduce")
      is_barrier_ = true;
    is_validate_to_trace_ = true;
    return name;
  };
  rebuild_cb_ = rebuild_fn;
  hang_counter_ = 0;
  event_ready_counter_ = 0;
  nvidia_launch_info_.duration_ = 0;
  may_ready = false;
}

void NvidiaGpuTimer::reset(
    cudaStream_t s, std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb,
    const std::string_view& type) {
  reset_stream(s, type);
  reset_cb(cb);
}

void NvidiaGpuTimer::reset(
    std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)> cb,
    const std::string_view& type) {
  is_host_ = true;
  type_ = type;
  launch_time_ = std::chrono::system_clock::now();
  reset_cb(cb);
}

void NvidiaGpuTimer::reset_stream(cudaStream_t s,
                                  const std::string_view& type) {
  is_host_ = false;
  stream_ = s;
  // assign is light, capture and copy object into closure is heavier.
  type_ = type;
  // record event of kernel launch
  startRecord();
  // get timestamp of kernel launch
  launch_time_ = std::chrono::system_clock::now();
  launch_time_for_hang_ = std::chrono::system_clock::now();
}

void NvidiaGpuTimer::reset(cudaStream_t s, const std::string_view& type,
                           const std::initializer_list<int64_t>& bmnk,
                           const std::initializer_list<int64_t>& ld,
                           const std::initializer_list<int64_t>& stride,
                           const int trans_a, const int trans_b, const int algo,
                           const std::string&& name_prefix,
                           cudaDataType_t dtype, const std::string& api,
                           uint8_t bias) {
  reset_stream(s, type);
  inner_rebuild_cb_.reset(bmnk, ld, stride, trans_a, trans_b, algo,
                          std::move(name_prefix), dtype, api, bias);
  reset_cb(inner_rebuild_cb_);
}

bool NvidiaGpuTimer::isValidateToTrace() { return is_validate_to_trace_; }

/*
 * ===================================
 * Class impl of NvidiaGpuTimer::MatmulBuilderCallback
 * ===================================
 */

NvidiaGpuTimer::FnReturn NvidiaGpuTimer::MatmulBuilderCallback::operator()(
    NvidiaGpuTimer* timer) {
  std::ostringstream oss;
  std::string compute_dtype = CudaDataTypeUtils::getCudaDtype(dtype_);
  timer->trace->Clear();
  timer->trace->set_kernel_type(constant::Metrics::MatmulMetrics::KERNEL_TYPE);

  hook::KernelDebugData* debug_data = timer->trace->mutable_debug_data();

  hook::NvidiaLaunchData* launch_data =
      debug_data->mutable_nvidia_launch_data();
  hook::MatmulDebugData* kernel_data = debug_data->mutable_mm_debug();

  launch_data->add_grids(timer->nvidia_launch_info_.grid_dim_.x);
  launch_data->add_grids(timer->nvidia_launch_info_.grid_dim_.y);
  launch_data->add_grids(timer->nvidia_launch_info_.grid_dim_.z);

  launch_data->add_blocks(timer->nvidia_launch_info_.block_dim_.x);
  launch_data->add_blocks(timer->nvidia_launch_info_.block_dim_.y);
  launch_data->add_blocks(timer->nvidia_launch_info_.block_dim_.z);

  launch_data->set_stream_id(timer->nvidia_launch_info_.stream_id_);

  kernel_data->set_dtype(compute_dtype);

  oss << name_prefix_;
  uint64_t flop = 2;
  for (const auto& v : bmnk_) {
    oss << v << "_";
    kernel_data->add_shapes(v);
    flop = flop * v;
  }
  *kernel_data->mutable_lds() = {ld_.begin(), ld_.end()};
  *kernel_data->mutable_strides() = {stride_.begin(), stride_.end()};
  kernel_data->set_algo(algo_);
  kernel_data->set_api(api_);
  kernel_data->set_trans((trans_a_ ? std::string("T") : std::string("N")) +
                         (trans_b_ ? std::string("T") : std::string("N")));

  return std::make_tuple(
      oss.str(), flop,
      xpu_timer::Labels{{"dtype", compute_dtype}, {"operation", "Matmul"}});
}

void NvidiaGpuTimer::MatmulBuilderCallback::reset(
    const std::initializer_list<int64_t>& bmnk,
    const std::initializer_list<int64_t>& ld,
    const std::initializer_list<int64_t>& stride, const int trans_a,
    const int trans_b, const int algo, const std::string&& name_prefix,
    cudaDataType_t dtype, const std::string& api, uint8_t bias) {
  std::copy(bmnk.begin(), bmnk.end(), bmnk_.begin());
  std::copy(ld.begin(), ld.end(), ld_.begin());
  std::copy(stride.begin(), stride.end(), stride_.begin());
  trans_a_ = trans_a;
  trans_b_ = trans_b;
  algo_ = algo;
  api_ = api;
  name_prefix_ = std::move(name_prefix);
  dtype_ = dtype;
  bias_ = bias;
}

#if defined(XPU_NVIDIA)
std::string NcclCommWrapper::getNcclVersion() {
  std::string nccl_lib_path =
      ::xpu_timer::util::config::GlobalConfig::dlopen_path["NCCL"];
  std::string nccl_version_cmd =
      "strings " + nccl_lib_path +
      R"(| grep -E "version.*cuda" | awk -F"[ +]" '{print $1 "_" $3}')";
  bp::environment env = boost::this_process::environment();
  env.erase("LD_PRELOAD");
  bp::ipstream out_stream;
  bp::ipstream err_stream;

  bp::child nccl_version_cmd_exec(
      "/bin/bash", bp::args({"-c", nccl_version_cmd}), env,
      bp::std_out > out_stream, bp::std_err > err_stream);

  std::string nccl_version;
  std::thread reader_stdout([&out_stream, &nccl_version] {
    std::string line;
    while (std::getline(out_stream, line)) {
      XLOG(INFO) << line;
      nccl_version = line;
    }
  });

  std::thread reader_stderr([&err_stream] {
    std::string line;
    while (std::getline(err_stream, line)) XLOG(INFO) << line;
  });

  nccl_version_cmd_exec.wait();
  out_stream.pipe().close();
  err_stream.pipe().close();
  reader_stderr.join();
  reader_stdout.join();
  int exit_code = nccl_version_cmd_exec.exit_code();
  if (exit_code) {
    XLOG(WARNING) << "Call nccl_version_cmd_exec error, code is " << exit_code
                  << " cmd is " << nccl_version_cmd;
    return "NCCL_NOT_FOUND";
  }
  XLOG(INFO) << "nccl version is " << nccl_version << " , cmd is "
             << nccl_version_cmd;
  return nccl_version;
}

// In NVIDIA, the address of ncclComm_t->devComm is used as the key, whereas in
// HPU, the value of ncllComm_t->devComm is used as the key. Therefore, we need
// two distinct constructor functions to differentiate between these cases.
void* NcclCommWrapper::handle = nullptr;

#define DEF_COMM_INFO_FUNCTION(FIELD, TYPE) \
  NcclCommWrapper::get_Comm_##FIELD##_Fn    \
      NcclCommWrapper::get_Comm_##FIELD##_func = nullptr;

DEF_COMM_INFO_FUNCTION(commHash, uint64_t)
DEF_COMM_INFO_FUNCTION(rank, int)
DEF_COMM_INFO_FUNCTION(nRanks, int)
DEF_COMM_INFO_FUNCTION(nNodes, int)
DEF_COMM_INFO_FUNCTION(devComm, void*)
#undef DEF_COMM_INFO_FUNCTION

void NcclCommWrapper::registerFunction() {
  static std::string nccl_version = getNcclVersion();
  std::string lib_path = "libparse_params.so";
  handle = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "cannot open library: " << dlerror() << std::endl;
    return;
  }
#define REGISTER_FUNCTION_TO_GET_COMM_INFO(FIELD)                    \
  get_Comm_##FIELD##_func = (get_Comm_##FIELD##_Fn)dlvsym(           \
      handle, "get_Comm_" #FIELD, nccl_version.c_str());             \
  if (!get_Comm_##FIELD##_func) {                                    \
    XLOG(WARNING) << "Not Found Func: get_Comm_" << #FIELD << " in " \
                  << nccl_version;                                   \
  } else {                                                           \
    XLOG(INFO) << "read symbols: get_Comm_" << #FIELD << " in "      \
               << nccl_version;                                      \
  }
  REGISTER_FUNCTION_TO_GET_COMM_INFO(commHash)
  REGISTER_FUNCTION_TO_GET_COMM_INFO(rank)
  REGISTER_FUNCTION_TO_GET_COMM_INFO(nRanks)
  REGISTER_FUNCTION_TO_GET_COMM_INFO(nNodes)
  REGISTER_FUNCTION_TO_GET_COMM_INFO(devComm)
#undef REGISTER_FUNCTION_TO_GET_COMM_INFO
}

NcclCommWrapper::NcclCommWrapper(ncclComm_t comm) {
  static std::once_flag registerFlag;
  std::call_once(registerFlag, []() { registerFunction(); });

#define GET_COMM_INFO(FIELD)               \
  if (get_Comm_##FIELD##_func) {           \
    FIELD = get_Comm_##FIELD##_func(comm); \
  }
  GET_COMM_INFO(commHash)
  GET_COMM_INFO(rank)
  GET_COMM_INFO(nRanks)
  GET_COMM_INFO(nNodes)
  GET_COMM_INFO(devComm)
#undef GET_COMM_INFO
}
#elif defined(XPU_NPU)
NcclCommWrapper::NcclCommWrapper(ncclComm_t comm)
    : commHash(comm->commHash),
      rank(comm->rank),
      nRanks(comm->nRanks),
      nNodes(comm->nNodes),
      devComm(comm->devComm) {}
#endif

}  // namespace nvidia
}  // namespace xpu_timer
