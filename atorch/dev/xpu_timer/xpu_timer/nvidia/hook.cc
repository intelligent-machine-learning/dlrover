#include "xpu_timer/nvidia/hook.h"

#include <cxxabi.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "xpu_timer/common/manager.h"
#include "xpu_timer/nvidia/nvidia_syms.h"

namespace atorch {

bool findNCCLSymsOrLaunch(std::string* s, cudaError_t* status,
                          cudaLaunchKernelExCFn fn,
                          const cudaLaunchConfig_t* config, const void* func,
                          void** args) {
  if (fns_to_skip.find(func) != fns_to_skip.end()) {
    *status = orig_cudaLaunchKernelExC(config, func, args);
    return false;
  }
  if (fns_to_name.find(func) == fns_to_name.end()) {
    ptrdiff_t offset = get_offset(func);
    if (addr_to_name.find(offset) != addr_to_name.end()) {
      fns_to_name[func] = addr_to_name[offset];
    } else {
      fns_to_skip.insert(func);
    }
  }
  *s = fns_to_name[func];
  return true;
}

ptrdiff_t get_offset(const void* symbol) {
  Dl_info info;
  if (dladdr(symbol, &info) != 0) {
    return (char*)symbol - (char*)info.dli_fbase;
  }
  return 0;
}

void resetNcclSymsMap() {
  const char* env_var = std::getenv("XPU_TIMER_LIB_PATH");
  if (env_var == nullptr) {
    return;
  }

  std::ostringstream oss;
  oss << "nm " << env_var
      << " | grep ncclKernel | grep -v __device_stub__ | c++filt | "
      << R"(awk 'BEGIN{a=""} /ncclKernel_/{fn=$3;sub(/\(.*/, "", fn);a = "0x"$1 "," fn "\n" a} END{print a}')";
  std::string nm_output = util::execShellCommand(oss.str().c_str());
  addr_to_name.clear();
  for (auto& each_line : util::split(nm_output, "\n")) {
    if (each_line.empty()) continue;
    std::vector<std::string> tokens = util::split(each_line, ",");
    ptrdiff_t addr = std::stoll(tokens[0], nullptr, 16);  // hex
    std::string func_name = tokens[1];
    addr_to_name[addr] = func_name;
    oss.str("");
    oss.clear();
    oss << std::hex << addr;
    LOG(INFO) << "read symbols " << oss.str() << ":" << func_name;
  }
}

void NvidiaGpuTimer::doPrepare() { resetNcclSymsMap(); }
void NvidiaGpuTimer::startRecord() { cudaEventRecord(startEvent_, stream_); }
void NvidiaGpuTimer::endRecord() { cudaEventRecord(stopEvent_, stream_); }

bool NvidiaGpuTimer::isReady() {
  return cudaEventQuery(stopEvent_) != cudaErrorNotReady;
}

uint64_t NvidiaGpuTimer::getDuration() {
  float elapsedTime;  // ms
  cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_);
  return uint64_t(elapsedTime * 1000);  // ms -> us
}

const std::string NvidiaGpuTimer::getName() { return description_(); }
void NvidiaGpuTimer::reset(cudaStream_t s,
                           std::function<const std::string()> des,
                           const std::string& type, uint64_t flop) {
  stream_ = s;
  description_ = des;
  type_ = type;
  flop_ = flop;
  startRecord();
}

const std::string NvidiaGpuTimer::getType() { return type_; }
const std::string NvidiaGpuTimer::getFlop() { return std::to_string(flop_); }

std::string demangle(const char* mangledName) {
  int status = -1;
  char* demangled = abi::__cxa_demangle(mangledName, NULL, NULL, &status);
  std::string result = (status == 0) ? demangled : mangledName;
  free(demangled);
  return result;
}

}  // namespace atorch

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream) {
  if (!orig_cudaLaunchKernel) {
    orig_cudaLaunchKernel =
        (cudaLaunchKernelFn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if (!orig_cudaLaunchKernel) {
      // nccl only use cudaLaunchKernelExC, maybe cudaLaunchKernel for other
      // kernels
    }
  }
  return orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                               stream);
}

EXPOSE_API
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                const void* func, void** args) {
  SETUP_DLSYM(cudaLaunchKernelExC);
  cudaError_t status;
  std::string name;

  if (!atorch::findNCCLSymsOrLaunch(&name, &status, orig_cudaLaunchKernelExC,
                                    config, func, args))
    return status;
  dim3 block_dim = config->blockDim;
  dim3 grid_dim = config->gridDim;
  auto fn = [name, block_dim, grid_dim]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_" << name << "__grid[" << grid_dim.x << "," << grid_dim.y
        << "," << grid_dim.z << "]block[" << block_dim.x << "," << block_dim.y
        << "," << block_dim.z << "]";
    return oss.str();
  };
  auto event =
      atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().getEvent();
  event->reset(config->stream, fn, "coll", 0);
  status = orig_cudaLaunchKernelExC(config, func, args);
  atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().recordEvent(
      event);
  return status;
}

EXPOSE_API
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType_t Atype,
    int lda, long long int strideA, const void* B, cudaDataType_t Btype,
    int ldb, long long int strideB, const void* beta, void* C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  SETUP_DLSYM(cublasGemmStridedBatchedEx);
  SETUP_DLSYM(cublasGetStream_v2);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto fn = [batchCount, m, n, k]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_bmm"
        << "_bmnk"
        << "_" << batchCount << "_" << m << "_" << n << "_" << k;
    return oss.str();
  };
  auto event =
      atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().getEvent();
  uint64_t flop = 2;
  event->reset(s, fn, "bmm", flop * batchCount * m * n * k);
  auto status = orig_cublasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype,
      ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType,
      algo);
  atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().recordEvent(
      event);
  return status;
}

EXPOSE_API
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A,
                            cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta,
                            void* C, cudaDataType Ctype, int ldc,
                            cudaDataType computeType, cublasGemmAlgo_t algo) {
  SETUP_DLSYM(cublasGemmEx);
  SETUP_DLSYM(cublasGetStream_v2);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto fn = [m, n, k]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_mm"
        << "_mnk"
        << "_" << m << "_" << n << "_" << k;
    return oss.str();
  };
  uint64_t flop = 2;
  auto event =
      atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().getEvent();
  event->reset(s, fn, "mm", flop * m * n * k);
  auto status =
      orig_cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                        B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  atorch::GpuTimerManager<atorch::NvidiaGpuTimer>::getInstance().recordEvent(
      event);
  return status;
}

#ifdef __cplusplus
}
#endif
