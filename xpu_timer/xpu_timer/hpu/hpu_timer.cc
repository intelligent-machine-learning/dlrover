#include "xpu_timer/hpu/hpu_timer.h"

#include <opdev/common_types.h>

#include <cstdlib>
#include <string>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"

namespace xpu_timer {
namespace hpu {

/*
 * ===================================
 * Class impl of EventStartTimeHelper
 * ===================================
 */

EventStartTimeHelper::EventStartTimeHelper(aclrtStream s) : stream_(s) {
  int rank = util::EnvVarRegistry::GetEnvVar<int>("LOCAL_RANK");
  aclrtSetDevice(rank);
  auto status = aclrtCreateEvent(&start_event_);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "create event err: " << aclGetRecentErrMsg();
  }
}

void EventStartTimeHelper::reset() {
  auto status = aclrtRecordEvent(start_event_, stream_);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "record event err: " << aclGetRecentErrMsg()
                << " stream: " << stream_;
  }
  status = aclrtSynchronizeEvent(start_event_);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "sync event err: " << aclGetRecentErrMsg()
                << " stream: " << stream_;
  }
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();
  cpu_time_ = std::chrono::duration_cast<std::chrono::microseconds>(
                  start.time_since_epoch())
                  .count();
}

time_t EventStartTimeHelper::getTime(aclrtEvent start_launch_event,
                                     bool* is_validate_to_trace) {
  float elapsed_time;  // ms
  aclrtEventElapsedTime(&elapsed_time, start_event_, start_launch_event);
  double es = ((double)elapsed_time * 1000);  // ms->us
  // Skip events that begin before the EventStartTimeHelper.
  *is_validate_to_trace = (es > 0);
  return cpu_time_ + time_t(es);
}

/*
 * ===================================
 * Class impl of HpuTimer
 * ===================================
 * ===================================
 * Static variables
 * ===================================
 */

std::unordered_map<uint64_t, uint64_t> HpuTimer::hccl_seq_num{};
std::unordered_map<std::string, int> HpuTimer::tracing_metas_{};
std::unordered_map<int, uint64_t> HpuTimer::trace_id_counter_{};
std::unordered_map<aclrtStream, EventStartTimeHelper*>
    HpuTimer::stream_timer_helper_{};
int HpuTimer::kernel_encoding_counter_(0);

/*
 * ===================================
 * Interface Overrides
 * ===================================
 * The following member functions are
 * overrides from the base interface.
 * ===================================
 */
const std::string_view& HpuTimer::getType() { return type_; }

const uint64_t HpuTimer::getProblemSize() { return problem_size_; }

void HpuTimer::startRecord() {
  auto status = aclrtRecordEvent(start_event_, stream_);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "record event err: " << aclGetRecentErrMsg()
                << " stream: " << stream_;
  }
}

void HpuTimer::endRecord() {
  if (is_host_) {
    finish_time_ = std::chrono::system_clock::now();
    return;
  }
  auto status = aclrtRecordEvent(stop_event_, stream_);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "record event err: " << aclGetRecentErrMsg()
                << " stream: " << stream_;
  }
}

int HpuTimer::getTraceCode() { return trace_code_; }

uint64_t HpuTimer::getTraceId() { return trace_id_; }

const std::string HpuTimer::getName() { return name_; }

bool HpuTimer::isReady() {
  if (is_host_) return true;
  aclrtEventRecordedStatus eventStatus;
  auto status = aclrtQueryEventStatus(stop_event_, &eventStatus);
  if (status != ACL_ERROR_NONE) {
    XLOG(ERROR) << "aclrtQueryEventStatus err: " << aclGetRecentErrMsg();
  }

  bool ready = (eventStatus == ACL_EVENT_RECORDED_STATUS_COMPLETE);
  if (!ready) {
    hang_counter_ += 1;
  } else {
    // After the HostEvent is ready, wait for the DeviceEvent
    // for 5 minutes, and also return ready=True. This is to prevent any
    // potential random hang issues.
    auto status = aclrtSynchronizeEventWithTimeout(stop_event_, 300000);
    if (status != ACL_ERROR_NONE) {
      XLOG(ERROR) << "aclrtSynchronizeEventWithTimeout err: "
                  << aclGetRecentErrMsg();
    }
  }
  return ready;
}

time_t HpuTimer::getExecuteTimeStamp() {
  auto execute_time = stream_timer_helper_[stream_]->getTime(
      start_event_, &(this->is_validate_to_trace_));
  return execute_time;
}

time_t HpuTimer::getLaunchTimeStamp() { return launch_time_timestamp_; }

uint64_t HpuTimer::getDuration() {
  if (is_host_) {
    auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
        finish_time_ - launch_time_);
    return dur_us.count();
  }
  float elapsed_time;  // ms
  auto status = aclrtEventElapsedTime(&elapsed_time, start_event_, stop_event_);
  if (status != ACL_ERROR_NONE) {
    elapsed_time = 0.0;
    XLOG(ERROR) << "aclrtEventElapsedTime event err: " << aclGetRecentErrMsg()
                << " stream: " << stream_;
  }
  return uint64_t(elapsed_time * 1000);  // ms -> us
}

void HpuTimer::reBuild() { name_ = rebuild_cb_(); }

Labels HpuTimer::getExtraLabels() { return extra_labels_; }

bool HpuTimer::isHang(time_t timeout) {
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

bool HpuTimer::ignoreHang() { return is_barrier_; }

bool HpuTimer::isHost() { return is_host_; }

/*
 * ===================================
 * Static Overload
 * ===================================
 * The following member functions are
 * overload from the base interface.
 * ===================================
 */

void HpuTimer::doPrepareForDumpTrace() {
  for (auto it : stream_timer_helper_) it.second->reset();
}

void HpuTimer::doPrepare() {
  std::string device_name_from_env =
      util::EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_DEVICE_NAME");

  // TODO(jingjun.lc): support device
  // if (device_name_from_env != util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
  //   XLOG(INFO) << "Device type set to " << device_name_from_env;
  //   HpuDataTypeUtils::setGpu(device_name_from_env);
  // } else {
  //   HpuDataTypeUtils::setGpu(platform::getDeviceName());
  // }
}

void HpuTimer::dumpTraceMeta(const std::string& path,
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

std::string HpuTimer::collBucketFn(double performance, uint64_t problem_size,
                                   Labels* label) {
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

std::string HpuTimer::matmulBucketFn(double peformance, uint64_t problem_size,
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

void HpuTimer::reset_cb(std::function<HpuTimer::FnReturn(HpuTimer*)> cb) {
  auto rebuild_fn = [this, cb]() -> auto {
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
      auto it = stream_timer_helper_.find(stream_);
      if (it == stream_timer_helper_.end()) {
        stream_timer_helper_.emplace(stream_,
                                     new EventStartTimeHelper(stream_));
      }
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
}

void HpuTimer::reset(aclrtStream s,
                     std::function<HpuTimer::FnReturn(HpuTimer*)> cb,
                     const std::string_view& type) {
  reset_stream(s, type);
  reset_cb(cb);
}

// void HpuTimer::reset(std::function<HpuTimer::FnReturn(HpuTimer*)> cb,
//                      const std::string_view& type) {
//   is_host_ = true;
//   type_ = type;
//   launch_time_ = std::chrono::system_clock::now();
//   reset_cb(cb);
// }

void HpuTimer::reset_stream(aclrtStream s, const std::string_view& type) {
  is_host_ = false;
  stream_ = s;
  // get timestamp of kernel launch
  launch_time_ = std::chrono::system_clock::now();
  launch_time_for_hang_ = std::chrono::system_clock::now();
  // record event of kernel launch
  startRecord();
  // assign is light, capture and copy object into closure is heavier.
  type_ = type;
}

bool HpuTimer::isValidateToTrace() { return is_validate_to_trace_; }

/*
 * ===================================
 * Class impl of InterceptManager
 * ===================================
 */
void InterceptManager::interceptMatmulInfo(const aclTensor* self,
                                           const aclTensor* other,
                                           aclOpExecutor** executor) {
  void* exe_addr = (void*)(*executor);
  int64_t* self_dims;
  uint64_t self_dims_num;
  aclDataType self_datatype;
  aclGetViewShape(self, &self_dims, &self_dims_num);
  aclGetDataType(self, &self_datatype);
  int64_t* other_dims;
  uint64_t other_dims_num;
  aclGetViewShape(other, &other_dims, &other_dims_num);

  auto get_bs_from_ND = [](uint64_t dims_num, int64_t* dims) -> auto {
    int64_t bs = 1;
    for (uint64_t i = 0; i < dims_num - 2; ++i) {
      bs *= dims[i];
    }
    return bs;
  };
  int64_t b = get_bs_from_ND(self_dims_num, self_dims);
  std::array<int64_t, 4> matmul_bmnk = {b, *(self_dims + self_dims_num - 2),
                                        *(other_dims + other_dims_num - 1),
                                        *(self_dims + self_dims_num - 1)};
  // XLOG(INFO) << "insert matmul info: " << exe_addr << " " << matmul_bmnk[0]
  //            << matmul_bmnk[1] << matmul_bmnk[2] << matmul_bmnk[3];
  matmul_info_map_[exe_addr] = std::make_shared<InterceptManager::MatmulInfo>(
      exe_addr, matmul_bmnk, self_datatype, "aclnnMatmul",
      "xpu_timer_aclnnMatmul_");
}

std::function<HpuTimer::FnReturn(HpuTimer*)> InterceptManager::handleMatmul(
    aclOpExecutor* executor) {
  auto matmul_info = [&]() -> auto {
    void* exe_addr = (void*)(executor);
    auto it = matmul_info_map_.find(exe_addr);
    if (it != matmul_info_map_.end()) {
      auto matmul_info_ptr = it->second;
      // XLOG(INFO) << "delete matmul info: " << exe_addr;
      // matmul_info_map_.erase(it);
      return matmul_info_ptr;
    } else {
      return std::make_shared<InterceptManager::MatmulInfo>(
          exe_addr, std::array<int64_t, 4>({1, 1, 1, 1}), ACL_FLOAT,
          "aclnnMatmul", "xpu_timer_aclnnMatmul_");
    }
  }();
  auto fn = [matmul_info](HpuTimer* timer) -> auto {
    std::ostringstream oss;
    timer->trace->Clear();
    timer->trace->set_kernel_type(
        constant::Metrics::MatmulMetrics::KERNEL_TYPE);

    hook::KernelDebugData* debug_data = timer->trace->mutable_debug_data();
    hook::MatmulDebugData* mm_debug = debug_data->mutable_mm_debug();

    std::string compute_dtype =
        HpuDataTypeUtils::getAclDtype(matmul_info->dtype_);
    mm_debug->set_dtype(compute_dtype);
    mm_debug->set_api("aclnnMatmul");

    oss << "aclnnMatmul";
    uint64_t flop = 2;
    for (const auto& v : matmul_info->bmnk_) {
      oss << v << "_";
      mm_debug->add_shapes(v);
      flop = flop * v;
    }

    return std::make_tuple(
        oss.str(), flop,
        xpu_timer::Labels{{"dtype", compute_dtype}, {"operation", "Matmul"}});
  };
  return fn;
}

void InterceptManager::interceptGroupedMatmulV2Info(
    const aclTensorList* x, const aclTensorList* weight,
    const aclIntArray* groupListOptional, int64_t splitItem, int64_t groupType,
    aclOpExecutor** executor) {
  void* exe_addr = (void*)(*executor);
  auto tensor_is_transpose = [](const aclTensor& tensor) -> auto {
    uint64_t strides_num;
    int64_t* strides_value;

    aclGetViewStrides(&tensor, &strides_value, &strides_num);
    return strides_value[strides_num - 1] != 1;
  };
  auto get_tensor_list_shape_fn =
      [&](const aclTensorList& tensor_list) -> auto {
    uint64_t tensor_list_size;
    aclGetTensorListSize(&tensor_list, &tensor_list_size);

    // {pair.first = dims number, pair.second = dims}
    std::vector<std::pair<uint64_t, int64_t*>> tensor_shape_list;
    for (uint64_t i = 0; i < tensor_list_size; ++i) {
      int64_t* dims;
      uint64_t dims_num;
      aclGetViewShape(tensor_list[i], &dims, &dims_num);
      const uint64_t LAST_2_DIM = dims_num - 2;
      const uint64_t LAST_DIM = dims_num - 1;
      if (tensor_is_transpose(*tensor_list[i])) {
        std::swap(dims[LAST_2_DIM], dims[LAST_DIM]);
      };
      tensor_shape_list.push_back({dims_num, dims});
    }
    return tensor_shape_list;
  };

  aclDataType datatype;
  aclGetDataType((*x)[0], &datatype);

  auto x_shape_list = get_tensor_list_shape_fn(*x);
  auto weight_shape_list = get_tensor_list_shape_fn(*weight);

  std::vector<std::array<int64_t, 4>> bmnks;
  auto get_bs_from_ND = [](uint64_t dims_num, int64_t* dims) -> auto {
    int64_t bs = 1;
    const uint64_t LAST_2_DIM = dims_num - 2;
    for (uint64_t i = 0; i < LAST_2_DIM; ++i) {
      bs *= dims[i];
    }
    return bs;
  };
  if (groupType == -1) {
    for (size_t i = 0; i < x_shape_list.size(); ++i) {
      const uint64_t x_dims_num = x_shape_list[i].first;
      const int64_t* x_dims = x_shape_list[i].second;
      const uint64_t weight_dims_num = weight_shape_list[i].first;
      const int64_t* weight_dims = weight_shape_list[i].second;
      const uint64_t x_LAST_2_DIM = x_dims_num - 2;
      const uint64_t x_LAST_DIM = x_dims_num - 1;
      const uint64_t weight_LAST_DIM = weight_dims_num - 1;

      const int64_t b =
          get_bs_from_ND(x_shape_list[i].first, x_shape_list[i].second);
      const int64_t m = x_dims[x_LAST_2_DIM];
      const int64_t k = x_dims[x_LAST_DIM];
      const int64_t n = weight_dims[weight_LAST_DIM];
      bmnks.push_back({b, m, n, k});
    }
  } else if (groupType == 0) {
    if (x_shape_list.size() > 1 && weight_shape_list.size() > 1 &&
        (splitItem == 2 || splitItem == 3)) {
      const uint64_t weight_dims_num = weight_shape_list[0].first;
      const int64_t* weight_dims = weight_shape_list[0].second;
      const uint64_t weight_LAST_DIM = weight_dims_num - 1;
      const int64_t n = weight_dims[weight_LAST_DIM];

      uint64_t groupListOptional_size;
      aclGetIntArraySize(groupListOptional, &groupListOptional_size);
      int64_t pre = 0;
      for (uint64_t i = 0; i < groupListOptional_size; ++i) {
        const uint64_t x_dims_num = x_shape_list[i].first;
        const int64_t* x_dims = x_shape_list[i].second;
        const uint64_t x_LAST_DIM = x_dims_num - 1;
        const int64_t k = x_dims[x_LAST_DIM];
        const int64_t m = (*groupListOptional)[i] - pre;
        pre = (*groupListOptional)[i];
        bmnks.push_back({1, m, n, k});
      }
    } else if (x_shape_list.size() == 1 && (splitItem == 2 || splitItem == 3)) {
      const uint64_t x_dims_num = x_shape_list[0].first;
      const int64_t* x_dims = x_shape_list[0].second;
      const uint64_t x_LAST_DIM = x_dims_num - 1;
      const int k = x_dims[x_LAST_DIM];
      const uint64_t weight_dims_num = weight_shape_list[0].first;
      const int64_t* weight_dims = weight_shape_list[0].second;
      const uint64_t weight_LAST_DIM = weight_dims_num - 1;
      const int n = weight_dims[weight_LAST_DIM];

      uint64_t groupListOptional_size;
      aclGetIntArraySize(groupListOptional, &groupListOptional_size);
      int64_t pre = 0;
      for (uint64_t i = 0; i < groupListOptional_size; ++i) {
        const int64_t m = (*groupListOptional)[i] - pre;
        pre = (*groupListOptional)[i];
        bmnks.push_back({1, m, n, k});
      }
    } else if (x_shape_list.size() == 1 && (splitItem == 0 || splitItem == 1)) {
      uint64_t groupListOptional_size;

      aclGetIntArraySize(groupListOptional, &groupListOptional_size);
      int64_t pre = 0;
      const uint64_t x_dims_num = x_shape_list[0].first;
      const int64_t* x_dims = x_shape_list[0].second;
      const uint64_t x_LAST_DIM = x_dims_num - 1;
      const int k = x_dims[x_LAST_DIM];

      for (uint64_t i = 0; i < groupListOptional_size; ++i) {
        const int64_t* weight_dims = weight_shape_list[i].second;
        const uint64_t weight_dims_num = weight_shape_list[i].first;
        const int64_t m = (*groupListOptional)[i] - pre;
        const uint64_t weight_LAST_DIM = weight_dims_num - 1;
        const int n = weight_dims[weight_LAST_DIM];
        pre = (*groupListOptional)[i];
        bmnks.push_back({1, m, n, k});
      }
    } else {
      XLOG(INFO) << "Unspported input arguments: x.shape.size = "
                 << x_shape_list.size()
                 << "weight.shape.size = " << weight_shape_list.size()
                 << " groupType " << groupType << " splitItem " << splitItem;
      bmnks.push_back({1, 1, 1, 1});
    }
  } else if (groupType == 2) {
    const uint64_t x_dims_num = x_shape_list[0].first;
    const int64_t* x_dims = x_shape_list[0].second;
    const uint64_t x_LAST_2_DIM = x_dims_num - 2;
    const int m = x_dims[x_LAST_2_DIM];
    const uint64_t weight_dims_num = weight_shape_list[0].first;
    const int64_t* weight_dims = weight_shape_list[0].second;
    const uint64_t weight_LAST_DIM = weight_dims_num - 1;
    const int n = weight_dims[weight_LAST_DIM];

    uint64_t groupListOptional_size;
    aclGetIntArraySize(groupListOptional, &groupListOptional_size);
    int64_t pre = 0;
    for (uint64_t i = 0; i < groupListOptional_size; ++i) {
      const int64_t k = (*groupListOptional)[i] - pre;
      pre = (*groupListOptional)[i];
      bmnks.push_back({1, m, n, k});
    }
  } else {
    XLOG(INFO) << "Unspported GroupType: groupType = " << groupType;
    bmnks.push_back({1, 1, 1, 1});
  }
  grouped_matmul_info_map_[exe_addr] =
      std::make_shared<InterceptManager::GroupedMamtulInfo>(
          exe_addr, bmnks, datatype, "aclnnGroupedMatmulV2",
          "xpu_timer_aclnnGroupedMatmulV2_");
}

std::function<HpuTimer::FnReturn(HpuTimer*)>
InterceptManager::handleGroupedMatmulV2(aclOpExecutor* executor) {
  auto grouped_matmul_info = [&]() -> auto {
    void* exe_addr = (void*)(executor);
    auto it = grouped_matmul_info_map_.find(exe_addr);
    if (it != grouped_matmul_info_map_.end()) {
      auto grouped_matmul_info_ptr = it->second;
      // grouped_matmul_info_map_.erase(it);
      return grouped_matmul_info_ptr;
    } else {
      return std::make_shared<InterceptManager::GroupedMamtulInfo>(
          exe_addr, std::vector({std::array<int64_t, 4>({1, 1, 1, 1})}),
          ACL_FLOAT, "aclnnGroupedMatmulV2", "xpu_timer_aclnnGroupedMatmulV2_");
    }
  }();
  auto fn = [grouped_matmul_info](HpuTimer* timer) -> auto {
    std::ostringstream oss;
    timer->trace->Clear();
    timer->trace->set_kernel_type(
        constant::Metrics::MatmulMetrics::KERNEL_TYPE);

    hook::KernelDebugData* debug_data = timer->trace->mutable_debug_data();
    hook::GroupedMatmulDebugData* grouped_mm_debug = debug_data->mutable_grouped_mm_debug();

    std::string compute_dtype =
        HpuDataTypeUtils::getAclDtype(grouped_matmul_info->dtype_);
    grouped_mm_debug->set_dtype(compute_dtype);
    grouped_mm_debug->set_api("aclnnGroupedMatmulV2");

    oss << "GroupedMatmul";

    uint64_t flop = 0;
    for (const auto& bmnk : grouped_matmul_info->bmnks_) {
      // hook::GroupedMatmulDebugData::BMNK* bmnks =
      // grouped_mm_debug->add_bmnks();
      uint64_t cur_flop = 2;
      for (const auto& v : bmnk) {
        // bmnks->add_shapes(v);
        cur_flop = cur_flop * v;
      }
      flop += cur_flop;
    }
    grouped_mm_debug->set_tflops(flop);

    return std::make_tuple(oss.str(), flop,
                           xpu_timer::Labels{{"dtype", compute_dtype},
                                             {"operation", "GroupedMatmul"}});
  };
  return fn;
}
std::function<HpuTimer::FnReturn(HpuTimer*)> InterceptManager::handleHccl(
    uint64_t count, HcclDataType datatype, HcclComm& comm,
    const std::string& func_name, const std::string& coll_type) {
  auto fn = [count, datatype, comm, func_name,
             coll_type](HpuTimer* timer) -> auto {
    std::ostringstream oss;
    timer->trace->Clear();
    timer->trace->set_kernel_type(constant::Metrics::CollMetrics::KERNEL_TYPE);
    hook::KernelDebugData* debug_data = timer->trace->mutable_debug_data();
    hook::NcclDebugData* hccl_debug = debug_data->mutable_nccl_debug();

    std::string dtype = HpuDataTypeUtils::getHcclDataType(datatype);
    uint64_t comm_size = count * HpuDataTypeUtils::getDtypeSizeInBytes(dtype);

    oss << "xpu_timer_" << func_name << "_size_" << comm_size;
    // cann-hccl/src/domain/collective_communication/framework/inc/topoinfo_struct.h
    constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH];
    HcclGetCommName(comm, commName);
    std::string commNameStr(commName);
    // TODO(jingjun): use global map to get commHash, only calculate once
    uint64_t commHash = std::hash<std::string>{}(commNameStr);

    auto HPU_CLUSTER_CARD_NUMBER = util::config::GlobalConfig::local_world_size;
    uint32_t nranks;
    HcclGetRankSize(comm, &nranks);
    uint32_t nNodes =
        (nranks + HPU_CLUSTER_CARD_NUMBER - 1) / HPU_CLUSTER_CARD_NUMBER;

    hccl_debug->set_comm_hash(commHash);
    hccl_debug->set_input_size_in_bytes(comm_size);
    hccl_debug->set_dtype(dtype);
    hccl_debug->set_ranks(nranks);
    hccl_debug->set_nodes(nNodes);
    hccl_debug->set_seq(++(timer->hccl_seq_num[commHash]));

    double factor = 1.0;
    if (coll_type == "AllReduce") {
      factor = 2.0 * (nranks - 1) / nranks;
    } else if (coll_type == "AllGather" || coll_type == "ReduceScatter") {
      // input of reduce_scatter/allgather is sharded, so we do not device
      // world_size
      factor = static_cast<double>(nranks - 1);
    }

    uint64_t problem_size = static_cast<uint64_t>(factor * comm_size);
    uint64_t problem_size_bits = problem_size * 8;

    hccl_debug->set_problem_size(problem_size);

    return std::make_tuple(
        oss.str(), problem_size_bits,
        xpu_timer::Labels{
            {"dtype", dtype},
            {"operation", coll_type},
            {"algorithm", "NotKown"},
            {"transport", nNodes > 1 ? "InterNode" : "IntraNode"}});
  };
  return fn;
}

}  // namespace hpu
}  // namespace xpu_timer
