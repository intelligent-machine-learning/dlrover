#pragma once
#include <bvar/bvar.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <functional>
#include <string>

#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/xpu_timer.h"

namespace atorch {

// get abs offset in shared object by address of function
ptrdiff_t get_offset(const void *symbol);

class NvidiaGpuTimer : public XpuTimer {
  /* Use cuda event to timing kernel.
   */
 public:
  explicit NvidiaGpuTimer() {
    cudaEventCreate(&startEvent_);
    cudaEventCreate(&stopEvent_);
  }
  // interfaces
  void startRecord() override;
  void endRecord() override;
  bool isReady() override;
  uint64_t getDuration() override;
  const std::string getName() override;
  const std::string getType() override;
  const std::string getFlop() override;

  // the event is in object pool, we reset it by different kernel.
  void reset(cudaStream_t s, std::function<const std::string()> des,
             const std::string &type, uint64_t flop);
  // parse nccl syms if needed.
  static void doPrepare();

 private:
  cudaEvent_t startEvent_, stopEvent_;  // owned
  cudaStream_t stream_;                 // not owned
  // return kernel name, it's callback function and called in background thread,
  // be careful of the lifetime of object in closure.
  std::function<const std::string()> description_;
  // kernel type, current is batched matmul, matmul, coll
  std::string type_;
  uint64_t flop_;
};

}  // namespace atorch

#ifdef __cplusplus
extern "C" {
#endif
typedef struct cublasContext *cublasHandle_t;
typedef enum {} cublasStatus_t;

typedef enum {} cublasOperation_t;

typedef enum {} cublasGemmAlgo_t;

typedef enum {} cublasComputeType_t;

typedef cudaError_t (*cudaLaunchKernelFn)(const void *, dim3, dim3, void **,
                                          size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchKernelExCFn)(const cudaLaunchConfig_t *,
                                             const void *, void **);
typedef cublasStatus_t (*cublasGemmExFn)(cublasHandle_t, cublasOperation_t,
                                         cublasOperation_t, int, int, int,
                                         const void *, const void *,
                                         cudaDataType, int, const void *,
                                         cudaDataType, int, const void *,
                                         void *, cudaDataType, int,
                                         cudaDataType, cublasGemmAlgo_t);
typedef cublasStatus_t (*cublasGetStream_v2Fn)(cublasHandle_t, cudaStream_t *);
typedef cublasStatus_t (*cublasGemmStridedBatchedExFn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void *, const void *, cudaDataType_t, int, long long int,
    const void *, cudaDataType_t, int, long long int, const void *, void *,
    cudaDataType_t, int, long long int, int, cublasComputeType_t,
    cublasGemmAlgo_t);

static cublasGemmStridedBatchedExFn orig_cublasGemmStridedBatchedEx = NULL;
static cublasGetStream_v2Fn orig_cublasGetStream_v2 = NULL;
static cudaLaunchKernelExCFn orig_cudaLaunchKernelExC = NULL;
static cudaLaunchKernelFn orig_cudaLaunchKernel = NULL;
static cublasGemmExFn orig_cublasGemmEx = NULL;
#ifdef __cplusplus
}
#endif
