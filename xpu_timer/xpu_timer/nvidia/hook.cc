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

#include "xpu_timer/nvidia/hook.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/manager.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/nvidia/nvidia_timer.h"

static void getMatrixDimensions(const cublasLtMatrixLayout_t& layout,
                                cudaDataType_t& dtype, int32_t& b,
                                uint64_t& rows, uint64_t& cols, uint64_t& ld,
                                int64_t& stride) {
  orig_cublasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b), NULL);
  orig_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS,
                                        &rows, sizeof(rows), NULL);
  orig_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_COLS,
                                        &cols, sizeof(cols), NULL);
  orig_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld,
                                        sizeof(ld), NULL);
  orig_cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                        &dtype, sizeof(dtype), NULL);
  orig_cublasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride), NULL);
}

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream) {
  SETUP_DLSYM(cudaLaunchKernel);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  const xpu_timer::nvidia::InterceptSymbol* sym;
  if (!xpu_timer::GpuTimerManager<
           xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
           .intercept_manager.isIntercepted(func, &sym)) {
    return orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  }
  bool skip_tp = false;
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
          .intercept_manager.handleCudaLaunchKernel(
              func, gridDim, blockDim, args, sharedMem, stream, sym, &skip_tp);
  if (skip_tp)
    return orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn,
               sym->func_type == "NCCL"
                   ? xpu_timer::constant::Metrics::CollMetrics::TYPE
                   : xpu_timer::constant::Metrics::MatmulMetrics::TYPE);
  cudaError_t status =
      orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

#if defined(CUDA_LAUNCH_EXC)
EXPOSE_API
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                const void* func, void** args) {
  SETUP_DLSYM(cudaLaunchKernelExC);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cudaLaunchKernelExC(config, func, args);
  cudaError_t status;
  const xpu_timer::nvidia::InterceptSymbol* sym;

  if (!xpu_timer::GpuTimerManager<
           xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
           .intercept_manager.isIntercepted(func, &sym)) {
    return orig_cudaLaunchKernelExC(config, func, args);
  }
  bool skip_tp = false;

  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.handleCudaLaunchKernelExC(config, func, args,
                                                             sym, &skip_tp);
  if (skip_tp) return orig_cudaLaunchKernelExC(config, func, args);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(config->stream, fn,
               xpu_timer::constant::Metrics::CollMetrics::TYPE);
  status = orig_cudaLaunchKernelExC(config, func, args);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}
#endif

EXPOSE_API
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType_t Atype,
    int lda, long long int strideA, const void* B, cudaDataType_t Btype,
    int ldb, long long int strideB, const void* beta, void* C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batch_count,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  SETUP_DLSYM_WITH_CUBLAS(cublasGemmStridedBatchedEx);
  SETUP_DLSYM_WITH_CUBLAS(cublasGetStream_v2);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B,
        Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count,
        computeType, algo);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_count, m, n, k}, {lda, ldb, ldc},
               {strideA, strideB, strideC}, static_cast<int>(transa),
               static_cast<int>(transb), static_cast<int>(algo),
               "xpu_timer_bmm_bias_bmnk_", Atype, "cublasGemmStridedBatchedEx");
  auto status = orig_cublasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype,
      ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count, computeType,
      algo);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
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
  SETUP_DLSYM_WITH_CUBLAS(cublasGemmEx);
  SETUP_DLSYM_WITH_CUBLAS(cublasGetStream_v2);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype,
                             lda, B, Btype, ldb, beta, C, Ctype, ldc,
                             computeType, algo);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {1, m, n, k}, {lda, ldb, ldc}, {0, 0, 0},
               static_cast<int>(transa), static_cast<int>(transb),
               static_cast<int>(algo), "xpu_timer_mm_bmnk_", Atype,
               "cublasGemmEx");

  auto status =
      orig_cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                        B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C,
                           int ldc) {
  SETUP_DLSYM_WITH_CUBLAS(cublasSgemm);
  SETUP_DLSYM_WITH_CUBLAS(cublasGetStream_v2);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {1, m, n, k}, {lda, ldb, ldc}, {0, 0, 0},
               static_cast<int>(transa), static_cast<int>(transb), -1,
               "xpu_timer_mm_bmnk_", CUDA_R_32F, "cublasSgemm");

  auto status = orig_cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                 B, ldb, beta, C, ldc);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batch_count) {
  SETUP_DLSYM_WITH_CUBLAS(cublasSgemmStridedBatched);
  SETUP_DLSYM_WITH_CUBLAS(cublasGetStream_v2);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
        strideB, beta, C, ldc, strideC, batch_count);
  cudaStream_t s;
  orig_cublasGetStream_v2(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_count, m, n, k}, {lda, ldb, ldc},
               {strideA, strideB, strideC}, static_cast<int>(transa),
               static_cast<int>(transb), -1, "xpu_timer_bmm_bmnk_", CUDA_R_32F,
               "cublasSgemmStridedBatched");
  auto status = orig_cublasSgemmStridedBatched(
      handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
      beta, C, ldc, strideC, batch_count);

  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  SETUP_DLSYM_WITH_CUBLASLT(cublasLtMatmul);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B,
                               Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace,
                               workspaceSizeInBytes, stream);

  SETUP_DLSYM_WITH_CUBLASLT(cublasLtMatrixLayoutGetAttribute);
  SETUP_DLSYM_WITH_CUBLASLT(cublasLtMatmulAlgoConfigGetAttribute);
  SETUP_DLSYM_WITH_CUBLASLT(cublasLtMatmulDescGetAttribute);
  cudaDataType_t dtype_a, dtype_b, dtype_c;
  int32_t batch_a, batch_b, batch_c;
  uint64_t rows_a, rows_b, rows_c;
  uint64_t cols_a, cols_b, cols_c;
  uint64_t ld_a, ld_b, ld_c;
  int64_t stride_a, stride_b, stride_c;
  int32_t trans_a, trans_b, trans_c;
  getMatrixDimensions(Adesc, dtype_a, batch_a, rows_a, cols_a, ld_a, stride_a);
  getMatrixDimensions(Bdesc, dtype_b, batch_b, rows_b, cols_b, ld_b, stride_b);
  getMatrixDimensions(Cdesc, dtype_c, batch_c, rows_c, cols_c, ld_c, stride_c);
  orig_cublasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<cublasLtMatmulDescAttributes_t>(
          3) /*CUBLASLT_MATMUL_DESC_TRANSA*/,
      &trans_a, sizeof(trans_a), NULL);
  orig_cublasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<cublasLtMatmulDescAttributes_t>(
          4) /*CUBLASLT_MATMUL_DESC_TRANSB*/,
      &trans_b, sizeof(trans_b), NULL);
  orig_cublasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<cublasLtMatmulDescAttributes_t>(
          5) /*CUBLASLT_MATMUL_DESC_TRANSC*/,
      &trans_c, sizeof(trans_c), NULL);

  int m = trans_a ? cols_a : rows_a;
  int k = trans_a ? rows_a : cols_a;
  int n = trans_b ? rows_b : cols_b;

  int algo_id;
  size_t size_written;
  orig_cublasLtMatmulAlgoConfigGetAttribute(
      algo,
      static_cast<cublasLtMatmulAlgoConfigAttributes_t>(
          0) /*CUBLASLT_ALGO_CONFIG_ID*/,
      &algo_id, sizeof(algo_id), &size_written);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_a, m, n, k}, {ld_a, ld_b, ld_c},
               {stride_a, stride_b, stride_c}, trans_a, trans_b, algo_id,
               "xpu_timer_bmm_bias_bmnk_", dtype_a, "cublasLtMatmul",
               1 /*has bias*/);
  auto status = orig_cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc,
                                    B, Bdesc, beta, C, Cdesc, D, Ddesc, algo,
                                    workspace, workspaceSizeInBytes, stream);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclAllReduce);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          count, datatype, comm, stream);

  return orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                            stream);
}

EXPOSE_API
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclReduce);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm,
                           stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(count, datatype, comm, stream);

  return orig_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm,
                         stream);
}

EXPOSE_API
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclAllGather);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          sendcount, datatype, comm, stream);

  return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm,
                            stream);
}

EXPOSE_API
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclReduceScatter);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                  comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          recvcount, datatype, comm, stream);

  return orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                comm, stream);
}

EXPOSE_API
ncclResult_t ncclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclSend);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclSend(sendbuff, count, datatype, peer, comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(
          count, datatype, comm, stream,
          xpu_timer::nvidia::InterceptManager::SendRecvType::Send);

  return orig_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

EXPOSE_API
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclRecv);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclRecv(recvbuff, count, datatype, peer, comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(
          count, datatype, comm, stream,
          xpu_timer::nvidia::InterceptManager::SendRecvType::Recv);

  return orig_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

EXPOSE_API
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(ncclBroadcast);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          count, datatype, comm, stream);

  return orig_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                            stream);
}

EXPOSE_API
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
  SETUP_DLSYM(cudaFreeAsync);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaFreeAsync", 1, "", false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaFreeAsync(devPtr, stream);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  SETUP_DLSYM(cudaMemcpyAsync);
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
  // cudaMemcpyHostToHost = 0   Host -> Host
  // cudaMemcpyHostToDevice = 1   Host -> Device
  // cudaMemcpyDeviceToHost = 2   Device -> Host
  // cudaMemcpyDeviceToDevice = 3   Device -> Device
  static std::vector<std::string> copy_kind{"H2H", "H2D", "D2H", "D2D",
                                            "Unkonwn"};
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMemcpyAsync", count,
                                                copy_kind[kind], false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMemcpyAsync(dst, src, count, kind, stream);

  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMalloc(void** devPtr, size_t size) {
  SETUP_DLSYM(cudaMalloc);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMalloc", size, "", true);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMalloc(devPtr, size);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaFree(void* devPtr) {
  SETUP_DLSYM(cudaFree);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaFree", 1, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaFree(devPtr);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size,
                                    cudaMemPool_t memPool,
                                    cudaStream_t stream) {
  SETUP_DLSYM(cudaMallocFromPoolAsync);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMallocFromPoolAsync", size,
                                                "", false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMallocFromPoolAsync(ptr, size, memPool, stream);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  SETUP_DLSYM(cudaHostAlloc);
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
          .intercept_manager.deviceMemory("cudaHostAlloc", size, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaHostAlloc(ptr, size, flags);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMallocHost(void** ptr, size_t size) {
  SETUP_DLSYM(cudaMallocHost);
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
          .intercept_manager.deviceMemory("cudaMallocHost", size, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMallocHost(ptr, size);
  xpu_timer::GpuTimerManager<xpu_timer::nvidia::NvidiaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

#ifdef __cplusplus
}
#endif
