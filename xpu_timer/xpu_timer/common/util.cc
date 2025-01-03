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

#include "xpu_timer/common/util.h"

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/err.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/version.h"

namespace xpu_timer {
namespace util {

namespace detail {

void ShmSwitch::reset() {
  reset(std::string(constant::KernelTraceConstant::DEFAULT_TRACE_DUMP_PATH),
        std::string(constant::KernelTraceConstant::DEFAULT_TRACE_DUMP_PATH),
        constant::KernelTraceConstant::DEFAULT_TRACE_COUNT, 0, 0, false);
}

void ShmSwitch::reset(const std::string& path, const std::string& oss_args,
                      uint32_t count, int64_t stamp, uint32_t dump_kernel) {
  uint32_t dump_path_length = path.length();
  std::strncpy(dump_path, path.data(), dump_path_length);
  dump_path[dump_path_length] = '\0';

  uint32_t oss_args_length = oss_args.length();
  std::strncpy(oss_dump_args, oss_args.data(), oss_args_length);
  oss_dump_args[oss_args_length] = '\0';

  dump_count = count;
  start_dump = 0;
  timestamp = stamp;
  dump_kernel_type = dump_kernel;
}

void ShmSwitch::reset(const std::string& path, const std::string& oss_args,
                      uint32_t count, int64_t stamp, uint32_t dump_kernel,
                      bool reset_signal) {
  reset(path, oss_args, count, stamp, dump_kernel);
  reset_flag = reset_signal;
}

InterProcessBarrierImpl::InterProcessBarrierImpl(std::string name,
                                                 int world_size, int rank)
    : name_(name) {
  bip::managed_shared_memory managed_shm(bip::open_or_create, name.c_str(),
                                         4096);  // one page is enough
  LOG(INFO) << "Barrier name in shm is " << name;
  LOG(INFO) << "World size " << world_size;
  InterProcessBarrierImpl::Inner*
      barriers[world_size];  // world size is small, allocate on stack is safe
  std::string barrier_name("InterProcessBarrierImpl" + std::to_string(rank));
  auto this_bar =
      managed_shm.find<InterProcessBarrierImpl::Inner>(barrier_name.c_str());
  if (this_bar.first) {
    barriers[rank] = this_bar.first;
  } else {
    barriers[rank] = managed_shm.construct<InterProcessBarrierImpl::Inner>(
        barrier_name.c_str())(false);
  }
  int index = 0;
  uint64_t try_count = 0;
  while (index < world_size) {
    std::string name = "InterProcessBarrierImpl" + std::to_string(index);
    auto this_bar =
        managed_shm.find<InterProcessBarrierImpl::Inner>(name.c_str());
    if (this_bar.first) {
      barriers[index] = this_bar.first;
      LOG(INFO) << "rank " << rank << " found index is " << index;
      index++;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      try_count++;
      if (try_count > 10000 * 10) {
        LOG(INFO) << "Rank " << rank << " waiting 10s for rank " << index
                  << " to create barrier obj in " << name_;
        try_count = 0;
      }
    }
  }
  // reset all state
  for (auto barrier : barriers) barrier->reset(false);
  try_count = 0;
  bool ready = false;
  // clang-format off
	// https://www.boost.org/doc/libs/1_84_0/doc/html/interprocess/some_basic_explanations.html#interprocess.some_basic_explanations.persistence
	// Note on Shared Memory Cleanup:
	// In Boost 1.84.0, shared memory resources (kernel or filesystem level) may not be correctly cleaned up 
	// if a process exits unexpectedly, despite utilizing RAII for resource management. This can lead to 
	// scenarios where shared memory resources are not properly initialized upon subsequent process startups.
	//
	// Solution for Barrier Synchronization:
	// To address potential initialization issues, we implement eventual consistency for barrier synchronization. 
	// Initially, all participating processes (ranks) reset their respective barrier flags to false before entering 
	// the barrier loop. Within this loop, each rank then independently marks itself as ready. This approach 
	// ensures robust barrier synchronization even if shared memory initialization is inconsistent.
  // clang-format on
  while (!ready) {
    ready = true;
    for (int i = 0; i < world_size; i++) {
      ready = barriers[i]->val && ready;
      if (try_count > 10000 && !barriers[i]->val) {
        LOG(INFO) << "Waiting rank " << i << " sleep 1s";
        try_count = 0;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    try_count++;
    // only set state in barrier operation
    barriers[rank]->reset(true);
  }
  LOG(INFO) << "Rank " << rank << " pass barrier " << name_;
}
InterProcessBarrierImpl::~InterProcessBarrierImpl() {
  bip::shared_memory_object::remove(name_.c_str());
}

}  // namespace detail

void InterProcessBarrier(int world_size, int rank, std::string name) {
  detail::InterProcessBarrierImpl(name, world_size, rank);
}

int ensureDirExists(const std::string& path) {
  std::filesystem::path dir_path(path);
  try {
    if (!std::filesystem::exists(dir_path)) {
      std::filesystem::create_directories(dir_path);
    }
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(ERROR) << "Create dir " << path << " error trace" << e.what();
    return 1;
  }
  return 0;
}

std::string execShellCommand(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  FILE* pipe = popen(cmd, "r");
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }
  int returnCode = pclose(pipe);
  if (returnCode != 0) {
    // TODO Handle error
    LOG(FATAL) << "Command failed with return code " << returnCode << std::endl;
  }
  return result;
}

// TODO change into native api when c++20 is ready.
std::vector<std::string> split(const std::string& str,
                               const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    tokens.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }
  tokens.push_back(str.substr(start, end));
  return tokens;
}

std::string base64_decode(const std::string& input) {
  BIO* bio = BIO_new_mem_buf(input.data(), input.size());
  BIO* b64 = BIO_new(BIO_f_base64());
  BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
  bio = BIO_push(b64, bio);

  std::string decoded_data(input.size(), '\0');
  int decoded_length = BIO_read(bio, &decoded_data[0], input.size());
  decoded_data.resize(decoded_length);

  BIO_free_all(bio);

  return decoded_data;
}

namespace detail {
struct EVP_CIPHER_CTX_Deleter {
  void operator()(EVP_CIPHER_CTX* ctx) const { EVP_CIPHER_CTX_free(ctx); }
};

};  // namespace detail

bool AES128_CBC(const std::string& ciphertext, std::string* text) {
  static const unsigned char key[16] = "xpu_timer";
  static const unsigned char iv[16] = "xpu_timer";
  const std::string& decode_ciphertext = base64_decode(ciphertext);
  std::unique_ptr<EVP_CIPHER_CTX, detail::EVP_CIPHER_CTX_Deleter> ctx(
      EVP_CIPHER_CTX_new());

  if (!ctx) return false;

  int len;
  int plaintext_len;

  std::vector<unsigned char> plaintext(decode_ciphertext.size());
  if (EVP_DecryptInit_ex(ctx.get(), EVP_aes_128_cbc(), NULL, key, iv) != 1)
    return false;

  if (EVP_DecryptUpdate(
          ctx.get(), plaintext.data(), &len,
          reinterpret_cast<const unsigned char*>(decode_ciphertext.data()),
          decode_ciphertext.size()) != 1)
    return false;
  plaintext_len = len;

  if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + len, &len) != 1)
    return false;
  plaintext_len += len;
  plaintext.resize(plaintext_len);
  *text =
      std::string(reinterpret_cast<char*>(plaintext.data()), plaintext.size());
  return true;
}

namespace config {
/*
 * ===================================
 * GlobalConfig
 * ===================================
 */
std::string GlobalConfig::pod_name("");
std::string GlobalConfig::ip("");
uint32_t GlobalConfig::rank{0};
uint32_t GlobalConfig::local_rank{0};
uint32_t GlobalConfig::world_size{0};
uint32_t GlobalConfig::local_world_size{0};
std::string GlobalConfig::job_name("");
std::string GlobalConfig::rank_str("");
bool GlobalConfig::enable{true};
std::vector<uint64_t> GlobalConfig::all_devices;
bool GlobalConfig::debug_mode{false};
std::unordered_map<std::string, std::string> GlobalConfig::dlopen_path;

/*
 * ===================================
 * BvarMetricsConfig
 * ===================================
 */

void setUpDlopenLibrary() {
  static bool has_set = false;
  if (has_set) return;
  std::ifstream maps_file("/proc/self/maps");
  bool nccl_found = false;
  bool cublas_lt_found = false;
  bool cublas_found = false;
  bool torch_cuda = false;

  std::string line;
  while (std::getline(maps_file, line)) {
    size_t pos = line.find('/');
    if (pos != std::string::npos) {
      std::string path = line.substr(pos);
      if (!nccl_found && path.find("libnccl.so") != std::string::npos) {
        GlobalConfig::dlopen_path["NCCL"] = path;
        nccl_found = true;
      } else if (!torch_cuda &&
                 path.find("libtorch_cuda.so") != std::string::npos) {
        GlobalConfig::dlopen_path["TORCH_CUDA"] = path;
        torch_cuda = true;
      } else if (!cublas_lt_found &&
                 path.find("libcublasLt.so") != std::string::npos) {
        GlobalConfig::dlopen_path["CUBLASLT"] = path;
        cublas_lt_found = true;
      } else if (!cublas_found &&
                 path.find("libcublas.so") != std::string::npos) {
        GlobalConfig::dlopen_path["CUBLAS"] = path;
        cublas_found = true;
      }
      if (nccl_found && cublas_lt_found && cublas_found && torch_cuda) {
        break;
      }
    }
  }

  // maybe static link
  if (!nccl_found)
    GlobalConfig::dlopen_path["NCCL"] = GlobalConfig::dlopen_path["TORCH_CUDA"];
  if (!cublas_lt_found)
    GlobalConfig::dlopen_path["CUBLASLT"] =
        GlobalConfig::dlopen_path["TORCH_CUDA"];
  if (!cublas_found)
    GlobalConfig::dlopen_path["CUBLAS"] =
        GlobalConfig::dlopen_path["TORCH_CUDA"];

  maps_file.close();
  for (const auto& lib : GlobalConfig::dlopen_path)
    LOG(INFO) << "[ENV] " << lib.first << " => " << lib.second;
  has_set = true;
}

time_t BvarMetricsConfig::bvar_window_size{0};
std::chrono::seconds BvarMetricsConfig::push_interval{0};
std::chrono::seconds BvarMetricsConfig::local_push_interval{0};
int BvarMetricsConfig::nic_bandwidth_gbps{0};
int BvarMetricsConfig::comm_bucket_count{0};
int BvarMetricsConfig::mm_bucket_count{0};
int BvarMetricsConfig::timeout_window_second{0};
std::map<std::string, std::string> BvarMetricsConfig::common_label;

void setUpConfig() {
  setUpGlobalConfig();
  setUpBvarMetricsConfig();
}

void setUpGlobalConfig() {
  GlobalConfig::pod_name = EnvVarRegistry::GetEnvVar<std::string>("POD_NAME");
  GlobalConfig::ip = EnvVarRegistry::GetEnvVar<std::string>("POD_IP");
  GlobalConfig::rank = EnvVarRegistry::GetEnvVar<int>("RANK");
  GlobalConfig::job_name =
      EnvVarRegistry::GetEnvVar<std::string>("ENV_ARGO_WORKFLOW_NAME");
  GlobalConfig::local_rank = EnvVarRegistry::GetEnvVar<int>("LOCAL_RANK");
  GlobalConfig::local_world_size =
      EnvVarRegistry::GetEnvVar<int>("LOCAL_WORLD_SIZE");
  GlobalConfig::world_size = EnvVarRegistry::GetEnvVar<int>("WORLD_SIZE");
  GlobalConfig::rank_str = "[RANK " + std::to_string(GlobalConfig::rank) + "] ";
  if (GlobalConfig::local_rank == 0) {
    LOG(INFO) << "[ENV] xpu_timer git version is " << git_version;
    LOG(INFO) << "[ENV] xpu_timer build time is " << build_time;
    LOG(INFO) << "[ENV] xpu_timer build type is " << build_type;
    LOG(INFO) << "[ENV] xpu_timer plantform is " << build_platform;
    LOG(INFO) << "[ENV] xpu_timer plantform version is "
              << build_platform_version;
  }
  GlobalConfig::debug_mode =
      EnvVarRegistry::GetEnvVar<bool>("XPU_TIMER_DEBUG_MODE");
#ifdef XPU_NVIDIA
  std::string dev_path = "/dev/nvidia";
#endif
  for (uint64_t device_index = 0; device_index < 16; device_index++) {
    std::filesystem::path dev(dev_path + std::to_string(device_index));
    if (std::filesystem::exists(dev)) {
      GlobalConfig::all_devices.push_back(device_index);
      if (GlobalConfig::local_rank == 0)
        LOG(INFO) << "[ENV] Found device " << dev;
    }
  }
  if (GlobalConfig::all_devices.empty()) GlobalConfig::enable = false;
  std::sort(GlobalConfig::all_devices.begin(), GlobalConfig::all_devices.end());

  if (GlobalConfig::local_world_size != GlobalConfig::all_devices.size()) {
    LOG(INFO) << "[ENV] local world size(" << GlobalConfig::local_world_size
              << ") is not equel to found devices("
              << GlobalConfig::all_devices.size() << ") disable hook";
    GlobalConfig::enable = false;
  }

  if (!GlobalConfig::enable)
    LOG(INFO) << "[ENV] Not all device are used, disable hook";
  if (GlobalConfig::debug_mode) {
    GlobalConfig::enable = true;
    LOG(INFO) << "[ENV] Debug mode is on, ignore all check";
  }
  setUpDlopenLibrary();
}

void setUpBvarMetricsConfig() {
  BvarMetricsConfig::push_interval = std::chrono::seconds(
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_PROMETHEUS_UPDATE_INTERVAL"));
  BvarMetricsConfig::local_push_interval = std::chrono::seconds(
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_LOCAL_UPDATE_INTERVAL"));
  BvarMetricsConfig::bvar_window_size =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_BVAR_WINDOW_SIZE");

  BvarMetricsConfig::common_label["pod_name"] = config::GlobalConfig::pod_name;
  BvarMetricsConfig::common_label["job_name"] = config::GlobalConfig::job_name;
  BvarMetricsConfig::common_label["ip"] = config::GlobalConfig::ip;
  BvarMetricsConfig::common_label["rank"] =
      std::to_string(config::GlobalConfig::rank);
  BvarMetricsConfig::common_label["local_rank"] =
      std::to_string(config::GlobalConfig::local_rank);
  BvarMetricsConfig::nic_bandwidth_gbps = 400;  // cx-7 is 400Gbps
  BvarMetricsConfig::comm_bucket_count =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_COMM_BUCKETING_COUNT");
  BvarMetricsConfig::mm_bucket_count =
      EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_MM_BUCKETING_COUNT");

  BvarMetricsConfig::timeout_window_second =
      util::EnvVarRegistry::GetEnvVar<int, false>(
          "XPU_TIMER_MM_BVAR_WINDOW_SIZE");
}

int getMinDeviceRank() {
  if (config::GlobalConfig::all_devices.empty()) return 0;
  return config::GlobalConfig::all_devices[0];
}
}  // namespace config

std::string getUniqueFileNameByCluster(const std::string& suffix) {
  std::ostringstream oss;
  std::string rank_str = std::to_string(config::GlobalConfig::rank);
  std::string world_size_str = std::to_string(config::GlobalConfig::world_size);
  size_t fill_rank = 5 - rank_str.length();
  size_t fill_world_size = 5 - world_size_str.length();
  oss << std::string(fill_rank, '0') << rank_str << "-"
      << std::string(fill_world_size, '0') << world_size_str << suffix;
  return oss.str();
}

void REGISTER_ENV() {
  REGISTER_ENV_VAR("POD_NAME", EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("POD_IP", EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("ENV_ARGO_WORKFLOW_NAME",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_CONFIG", EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_TIMELINE_PATH", "/root/timeline");
  REGISTER_ENV_VAR("XPU_TIMER_DAEMON_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_SYMS_FILE",
                   util::EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_LOGGING_DIR",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_LOGGING_APPEND", false);

  REGISTER_ENV_VAR("RANK", 0);
  REGISTER_ENV_VAR("LOCAL_RANK", 0);
  REGISTER_ENV_VAR("LOCAL_WORLD_SIZE", 0);
  REGISTER_ENV_VAR("WORLD_SIZE", 1);
  REGISTER_ENV_VAR("XPU_TIMER_TIMELINE_TRACE_COUNT",
                   constant::KernelTraceConstant::DEFAULT_TRACE_COUNT);
  REGISTER_ENV_VAR("XPU_TIMER_PROMETHEUS_UPDATE_INTERVAL", 10);  // second
  REGISTER_ENV_VAR("XPU_TIMER_LOCAL_UPDATE_INTERVAL", 1);
  REGISTER_ENV_VAR("XPU_TIMER_MM_BVAR_WINDOW_SIZE", 10);
  REGISTER_ENV_VAR("XPU_TIMER_COLL_BVAR_WINDOW_SIZE", 10);
  REGISTER_ENV_VAR("XPU_TIMER_DEREGISTER_TIME", 300);  // second
  REGISTER_ENV_VAR("XPU_TIMER_BUCKETING_METRICS", true);
  REGISTER_ENV_VAR("XPU_TIMER_MM_BUCKETING_COUNT", 10);
  REGISTER_ENV_VAR("XPU_TIMER_COMM_BUCKETING_COUNT", 20);

  REGISTER_ENV_VAR("XPU_TIMER_BASEPORT", 18888);
  REGISTER_ENV_VAR("XPU_TIMER_PORT", EnvVarRegistry::INT_DEFAULT_VALUE);

  REGISTER_ENV_VAR("XPU_TIMER_HANG_TIMEOUT", 300);  // second
  REGISTER_ENV_VAR("XPU_TIMER_HANG_KILL", false);

  REGISTER_ENV_VAR("XPU_TIMER_DUMP_STACK_COUNT", 0);
  REGISTER_ENV_VAR("XPU_TIMER_ALL_DUMP_TIMELINE", true);

  REGISTER_ENV_VAR("XPU_TIMER_DEBUG_MODE", false);
  REGISTER_ENV_VAR("XPU_TIMER_DEVICE_NAME",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_SKIP_TP", false);
  REGISTER_ENV_VAR("XPU_TIMER_TP_SIZE", 8);
  REGISTER_ENV_VAR("XPU_TIMER_HOST_TRACING_FUNC",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);

  // NCCL libs are built in framework, like PyTorch.
  REGISTER_ENV_VAR("XPU_TIMER_NCCL_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_TORCH_CUDA_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_CUBLAS_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_CUBLASLT_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_HPU_OPAPI_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);
  REGISTER_ENV_VAR("XPU_TIMER_HPU_HCCL_LIB_PATH",
                   EnvVarRegistry::STRING_DEFAULT_VALUE);

  // TODO(jingjun): Recreate the environment to diagnose the segmentation fault
  // occurring during matrix multiplication.
  REGISTER_ENV_VAR("XPU_TIMER_HPU_HOOK_MATMUL", false);
}

}  // namespace util
}  // namespace xpu_timer
