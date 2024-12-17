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
#include <butil/logging.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <variant>
#include <vector>

namespace bip = boost::interprocess;

namespace xpu_timer {
namespace util {
namespace config {
struct GlobalConfig {
  static std::string pod_name;
  static std::string ip;
  static uint32_t rank;
  static uint32_t local_rank;
  static uint32_t local_world_size;
  static uint32_t world_size;
  static std::string job_name;
  static std::string rank_str;
  static bool enable;
  static std::vector<uint64_t> all_devices;
  static bool debug_mode;
  static std::unordered_map<std::string, std::string> dlopen_path;
};

struct BvarMetricsConfig {
  static time_t bvar_window_size;
  static std::chrono::seconds push_interval;
  static std::chrono::seconds local_push_interval;
  static int nic_bandwidth_gbps;
  static std::map<std::string, std::string> common_label;
  static int comm_bucket_count;
  static int mm_bucket_count;
  static int timeout_window_second;
};

void setUpConfig();
void setUpGlobalConfig();
void setUpBvarMetricsConfig();
void setUpDlopenLibrary();
int getMinDeviceRank();

}  // namespace config
namespace detail {
struct InterProcessBarrierImpl {
  struct Inner {
    alignas(8) bool val;
    Inner(bool val) : val(val) {}
    void reset(bool value) { val = value; }
  };

  InterProcessBarrierImpl(std::string name, int world_size, int rank);
  ~InterProcessBarrierImpl();
  std::string name_;
};
}  // namespace detail

std::string getUniqueFileNameByCluster(const std::string& suffix);
void REGISTER_ENV();
void InterProcessBarrier(int world_size, int rank,
                         std::string name = "barrier");

std::string execShellCommand(const char* cmd);
std::vector<std::string> split(const std::string& str,
                               const std::string& delimiter);

bool AES128_CBC(const std::string& ciphertext, std::string* text);

class ScopeGuard {
 public:
  explicit ScopeGuard(std::function<void()> cb) : cb_(cb) {}

  ~ScopeGuard() { cb_(); }

 private:
  std::function<void()> cb_;
};

int ensureDirExists(const std::string& path);

namespace detail {
struct ShmSwitch {
  static constexpr std::string_view BarrierName = "shm_switch_barrier";
  static constexpr std::string_view ShmName = "ShmSwitch";
  static constexpr std::string_view ObjName = "ShmSwitchObj";
  alignas(8) char dump_path[1024];
  alignas(8) char oss_dump_args[4096];
  alignas(8) uint32_t dump_count;
  alignas(8) int start_dump;
  alignas(8) int64_t timestamp;
  alignas(8) bool reset_flag;
  alignas(8) uint32_t dump_kernel_type;  // in bits, 0 is matmul, 1 is comm
  void reset();
  void reset(const std::string& path, const std::string& oss_args,
             uint32_t count, int64_t stamp, uint32_t dump_kernel);
  void reset(const std::string& path, const std::string& oss_args,
             uint32_t count, int64_t stamp, uint32_t dump_kernel,
             bool reset_signal);
};
}  // namespace detail

template <typename T>
class ShmType {
 public:
  explicit ShmType(int local_world_size, int local_rank, bool main = true) {
    shm_name_ = std::string(T::ShmName);
    std::string obj_name(T::ObjName);
    size_t total_size = sizeof(T) * 4;

    std::string barrier_name = std::string(T::BarrierName);
    if (main) {
      bip::shared_memory_object::remove(shm_name_.c_str());
      shm_area_ = new bip::managed_shared_memory(bip::create_only,
                                                 shm_name_.c_str(), total_size);
      obj_ = shm_area_->construct<T>(obj_name.c_str())();
      obj_->reset();
      InterProcessBarrier(local_world_size, local_rank, barrier_name.c_str());
    } else {
      InterProcessBarrier(local_world_size, local_rank, barrier_name.c_str());
      shm_area_ =
          new bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
      auto find = shm_area_->find<T>(obj_name.c_str());
      if (find.first)
        obj_ = find.first;
      else {
        // never here
        LOG(INFO) << "rank " << local_rank << " do not found";
        std::abort();
      }
    }
  }
  ~ShmType() { bip::shared_memory_object::remove(shm_name_.c_str()); }

  T* getObj() { return obj_; }

 private:
  T* obj_;
  bip::managed_shared_memory* shm_area_;
  std::string shm_name_;
};

using ShmSwitch = ShmType<detail::ShmSwitch>;

template <typename T>
class BlockingDeque {
  /* Blocking deque, this is use for queuing XpuTimer object, backgroud thread
   * get object from the queue and parse the duration and name, then pushing to
   * prometheus. The working queue call push in launch kernel thread, and pop in
   * backgroud thread, usually, pop is more offen than push, so we let pop
   * blocking.
   */
 private:
  std::deque<T*> deque_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;

 public:
  void push(T* valuePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    deque_.push_back(valuePtr);
    cond_var_.notify_one();
    // cond_var_.notify_all(); // maybe, check thread nums.
  }

  T* pop(std::function<bool(T*)> is_hang, std::function<bool(T*)> is_ready,
         int* size) {
    std::unique_lock<std::mutex> lock(mutex_);
    *size = deque_.size();
    if (!cond_var_.wait_for(lock, std::chrono::seconds(5),
                            [this] { return !deque_.empty(); })) {
      LOG(INFO) << "No event to pop for 5 seconds, return null";
      return nullptr;
    }
    T* value_ptr = deque_.front();
    if (is_hang(value_ptr)) return value_ptr;
    if (!is_ready(value_ptr)) return nullptr;
    deque_.pop_front();
    return value_ptr;
  }

  void printHangName(std::vector<std::string>* hang_items) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it : deque_) {
      it->reBuild();
      LOG(INFO) << "Hang items " << it->getName();
      hang_items->push_back(it->getName());
    }
  }
};

template <typename T>
class TimerPool {
  /* This is a deque for pooling XpuTimer object.
   */
 public:
  TimerPool() = default;

  TimerPool(const TimerPool&) = delete;
  TimerPool& operator=(const TimerPool&) = delete;

  template <bool Create = true>
  T* getObject() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) {
      if constexpr (Create) return new T();
      return nullptr;
    } else {
      T* obj = pool_.front();
      pool_.pop_front();
      return obj;
    }
  }

  // Return an object to the pool
  void returnObject(T* obj, int* size) {
    *size = 0;
    if (obj) {
      std::lock_guard<std::mutex> lock(mutex_);
      pool_.push_back(obj);
      *size = pool_.size();
    }
  }

 private:
  std::deque<T*> pool_;
  std::mutex mutex_;
};

class EnvVarRegistry {
 public:
  using VarType = std::variant<int, bool, std::string>;
  static constexpr std::string_view STRING_DEFAULT_VALUE = "NOT_SET";
  static constexpr int INT_DEFAULT_VALUE = 0;
  static constexpr bool BOOL_DEFAULT_VALUE = false;

  static inline VarType convert_to_variant(const std::string_view& sv) {
    return std::string(sv);
  }

  static inline VarType convert_to_variant(const char* s) {
    return std::string(s);
  }

  template <typename T>
  static inline VarType convert_to_variant(const T& val) {
    return val;
  }

  static void RegisterEnvVar(const std::string& name, VarType default_value) {
    auto& registry = GetRegistry();
    std::string str_val = std::visit(
        [](const auto& value) -> std::string {
          std::stringstream ss;
          ss << value;
          return ss.str();
        },
        default_value);
    LOG(INFO) << "[ENV] Register ENV " << name << " with default " << str_val;
    registry[name] = default_value;
  }

  template <typename T, bool Print = true>
  static T GetEnvVar(const std::string& name) {
    auto& registry = GetRegistry();
    bool has_env = true;
    // if has register env, we get env as below
    // 1. return value if find in environment
    // 2. return value from config file
    // 3. return registered default value
    if (auto it = registry.find(name); it != registry.end()) {
      auto result = getEnvInner<T>(name, &has_env);
      if (has_env) {
        if constexpr (Print)
          LOG(INFO) << "[ENV] Get " << name << "=" << result
                    << " from environment";
        return result;
      }

      auto& pt = GetPtree();
      if (auto it = pt.find(name); it != pt.not_found()) {
        auto result = pt.get<T>(name);
        if constexpr (Print)
          LOG(INFO) << "[ENV] Get " << name << "=" << result << " from config";
        return result;
      }
      if (const T* result_p = std::get_if<T>(&it->second)) {
        if constexpr (Print)
          LOG(INFO) << "[ENV] Get " << name << "=" << *result_p
                    << " from register default";
        return *result_p;
      } else {
        // GetEnvVar is a internal api, so you should verify it, it not, we
        // abort
        if constexpr (Print)
          LOG(FATAL) << "[ENV] Wrong data type in `GetEnvVar`";
      }
    } else {
      auto result = getEnvInner<T>(name, &has_env);
      if (has_env) {
        if constexpr (Print)
          LOG(INFO) << "[ENV] Get " << name << "=" << result
                    << " from environment";
        return result;
      }
    }
    // if not register value, return default value for different dtype
    auto result = getDefault<T>();
    if constexpr (Print)
      LOG(WARNING) << "[ENV] Get not register env " << name << "=" << result
                   << " from default";
    return result;
  }

  static std::string getLibPath(const std::string& lib_name) {
    const std::string& env_name = "XPU_TIMER_" + lib_name + "_LIB_PATH";
    auto lib_path = GetEnvVar<std::string>(env_name);
    if (lib_path != STRING_DEFAULT_VALUE) {
      LOG(INFO) << "[ENV] Get lib path for dlopen " << lib_name << "="
                << lib_path << " from env " << env_name;
      return lib_path;
    }
    lib_path = config::GlobalConfig::dlopen_path[lib_name];
    if (lib_path.empty()) {
      std::cerr << "[ENV] Can't find any " << lib_name
                << " lib path from default" << std::endl;
      std::exit(1);
    }
    LOG(INFO) << "[ENV] Get lib path for dlopen " << lib_name << "=" << lib_path
              << " by default value. You can change it via env " << env_name;

    return lib_path;
  }

 private:
  template <typename T>
  static T getEnvInner(std::string env_name, bool* has_env) {
    const char* env = std::getenv(env_name.c_str());
    if (!env) {
      *has_env = false;
      return T{};
    }
    if constexpr (std::is_same_v<T, int>) {
      return std::atoi(env);
    } else if constexpr (std::is_same_v<T, bool>) {
      return std::atoi(env) != 0;
    } else if constexpr (std::is_same_v<T, std::string>) {
      return std::string(env);
    } else {
      static_assert(std::is_same_v<T, int> || std::is_same_v<T, bool> ||
                        std::is_same_v<T, std::string>,
                    "Unsupported type");
      return T{};  // never goes here
    }
  }

  template <typename T>
  static T getDefault() {
    if constexpr (std::is_same_v<T, int>) {
      return EnvVarRegistry::INT_DEFAULT_VALUE;
    } else if constexpr (std::is_same_v<T, bool>) {
      return EnvVarRegistry::BOOL_DEFAULT_VALUE;
    } else if constexpr (std::is_same_v<T, std::string>) {
      return std::string(EnvVarRegistry::STRING_DEFAULT_VALUE);
    } else {
      static_assert(std::is_same_v<T, int> || std::is_same_v<T, bool> ||
                        std::is_same_v<T, std::string>,
                    "Unsupported type");
      return T{};  // never goes here
    }
  }

  static std::unordered_map<std::string, VarType>& GetRegistry() {
    static std::unordered_map<std::string, VarType> registry;
    return registry;
  }

  static boost::property_tree::ptree& GetPtree() {
    static boost::property_tree::ptree pt;
    static bool pt_init_flag = false;

    if (!pt_init_flag) {
      pt_init_flag = true;
      const char* config_path = std::getenv("XPU_TIMER_CONFIG");
      if (config_path && config_path != EnvVarRegistry::STRING_DEFAULT_VALUE) {
        if (std::filesystem::exists(config_path))
          boost::property_tree::ini_parser::read_ini(config_path, pt);
        else
          LOG(WARNING) << "XPU_TIMER_CONFIG config " << config_path
                       << " is not exists, ignore it";
      }
    }
    return pt;
  }
};

#define REGISTER_ENV_VAR(name, value)                \
  ::xpu_timer::util::EnvVarRegistry::RegisterEnvVar( \
      name, ::xpu_timer::util::EnvVarRegistry::convert_to_variant(value))

}  // namespace util
}  // namespace xpu_timer
