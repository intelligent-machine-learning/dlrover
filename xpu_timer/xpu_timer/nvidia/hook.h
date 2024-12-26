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

#include <functional>
#include <string>

#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext* cublasHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;
// uint64 array, we use pointer to mock it
typedef uint64_t* cublasLtMatmulDesc_t;
// uint64 array, we use pointer to mock it
typedef uint64_t* cublasLtMatrixLayout_t;

typedef enum {} cublasLtMatmulAlgo_t;

typedef enum {} cublasStatus_t;

typedef enum {} cublasOperation_t;

typedef enum {} cublasGemmAlgo_t;

typedef enum {} cublasComputeType_t;

typedef enum {
  CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
  CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
  CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
  CUBLASLT_MATRIX_LAYOUT_COLS = 3,
  CUBLASLT_MATRIX_LAYOUT_LD = 4,
  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
  CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} cublasLtMatrixLayoutAttribute_t;

typedef enum {} cublasLtMatmulDescAttributes_t;

typedef enum {} cublasLtMatmulAlgoConfigAttributes_t;

typedef cudaError_t (*cudaLaunchKernelFn)(const void*, dim3, dim3, void**,
                                          size_t, cudaStream_t);
#if defined(CUDA_LAUNCH_EXC)
typedef cudaError_t (*cudaLaunchKernelExCFn)(const cudaLaunchConfig_t*,
                                             const void*, void**);
#endif
typedef cublasStatus_t (*cublasGemmExFn)(cublasHandle_t, cublasOperation_t,
                                         cublasOperation_t, int, int, int,
                                         const void*, const void*, cudaDataType,
                                         int, const void*, cudaDataType, int,
                                         const void*, void*, cudaDataType, int,
                                         cudaDataType, cublasGemmAlgo_t);
typedef cublasStatus_t (*cublasGetStream_v2Fn)(cublasHandle_t, cudaStream_t*);
typedef cublasStatus_t (*cublasGemmStridedBatchedExFn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, long long int, const void*,
    cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int,
    long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
typedef cublasStatus_t (*cublasSgemmFn)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);
typedef cublasStatus_t (*cublasSgemmStridedBatchedFn)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batchCount);

typedef cublasStatus_t (*cublasLtMatmulFn)(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);

typedef cublasStatus_t (*cublasLtMatrixLayoutGetAttributeFn)(
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef cublasStatus_t (*cublasLtMatmulAlgoConfigGetAttributeFn)(
    const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef cublasStatus_t (*cublasLtMatmulDescGetAttributeFn)(
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef ncclResult_t (*ncclAllReduceFn)(const void* sendbuff, void* recvbuff,
                                        size_t count, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm_t comm,
                                        cudaStream_t stream);
typedef ncclResult_t (*ncclReduceFn)(const void* sendbuff, void* recvbuff,
                                     size_t count, ncclDataType_t datatype,
                                     ncclRedOp_t op, int root, ncclComm_t comm,
                                     cudaStream_t stream);

typedef ncclResult_t (*ncclAllGatherFn)(const void* sendbuff, void* recvbuff,
                                        size_t sendcount,
                                        ncclDataType_t datatype,
                                        ncclComm_t comm, cudaStream_t stream);

typedef ncclResult_t (*ncclReduceScatterFn)(const void* sendbuff,
                                            void* recvbuff, size_t recvcount,
                                            ncclDataType_t datatype,
                                            ncclRedOp_t op, ncclComm_t comm,
                                            cudaStream_t stream);

typedef ncclResult_t (*ncclSendFn)(const void* sendbuff, size_t count,
                                   ncclDataType_t datatype, int peer,
                                   ncclComm_t comm, cudaStream_t stream);

typedef ncclResult_t (*ncclRecvFn)(void* recvbuff, size_t count,
                                   ncclDataType_t datatype, int peer,
                                   ncclComm_t comm, cudaStream_t stream);

typedef ncclResult_t (*ncclBroadcastFn)(const void* sendbuff, void* recvbuff,
                                        size_t count, ncclDataType_t datatype,
                                        int root, ncclComm_t comm,
                                        cudaStream_t stream);

typedef cudaError_t (*cudaMemcpyAsyncFn)(void* dst, const void* src,
                                         size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream);

typedef cudaError_t (*cudaFreeAsyncFn)(void* dst, cudaStream_t stream);

typedef cudaError_t (*cudaFreeFn)(void* devPtr);

typedef cudaError_t (*cudaMallocFn)(void** devPtr, size_t size);

typedef cudaError_t (*cudaMallocAsyncFn)(void** ptr, size_t size,
                                         cudaMemPool_t memPool,
                                         cudaStream_t stream);

typedef cudaError_t (*cudaMallocFromPoolAsyncFn)(void** ptr, size_t size,
                                                 cudaMemPool_t memPool,
                                                 cudaStream_t stream);
typedef cudaError_t (*cudaHostAllocFn)(void** ptr, size_t size,
                                       unsigned int flags);

typedef cudaError_t (*cudaMallocHostFn)(void** ptr, size_t size);

static cublasGemmStridedBatchedExFn orig_cublasGemmStridedBatchedEx = NULL;
static cublasGetStream_v2Fn orig_cublasGetStream_v2 = NULL;

#if defined(CUDA_LAUNCH_EXC)
static cudaLaunchKernelExCFn orig_cudaLaunchKernelExC = NULL;
#endif

static cudaLaunchKernelFn orig_cudaLaunchKernel = NULL;
static cublasGemmExFn orig_cublasGemmEx = NULL;
static cublasSgemmFn orig_cublasSgemm = NULL;
static cublasSgemmStridedBatchedFn orig_cublasSgemmStridedBatched = NULL;
static cublasLtMatmulFn orig_cublasLtMatmul = NULL;
static cublasLtMatrixLayoutGetAttributeFn
    orig_cublasLtMatrixLayoutGetAttribute = NULL;
static cublasLtMatmulAlgoConfigGetAttributeFn
    orig_cublasLtMatmulAlgoConfigGetAttribute = NULL;
static cublasLtMatmulDescGetAttributeFn orig_cublasLtMatmulDescGetAttribute =
    NULL;
static ncclAllReduceFn orig_ncclAllReduce = NULL;
static ncclReduceFn orig_ncclReduce = NULL;
static ncclAllGatherFn orig_ncclAllGather = NULL;
static ncclReduceScatterFn orig_ncclReduceScatter = NULL;
static ncclSendFn orig_ncclSend = NULL;
static ncclRecvFn orig_ncclRecv = NULL;
static ncclBroadcastFn orig_ncclBroadcast = NULL;

static cudaMemcpyAsyncFn orig_cudaMemcpyAsync = NULL;
static cudaFreeAsyncFn orig_cudaFreeAsync = NULL;
static cudaMallocAsyncFn orig_cudaMallocAsync = NULL;

static cudaFreeFn orig_cudaFree = NULL;
static cudaMallocFn orig_cudaMalloc = NULL;
static cudaMallocFromPoolAsyncFn orig_cudaMallocFromPoolAsync = NULL;
static cudaHostAllocFn orig_cudaHostAlloc = NULL;
static cudaMallocHostFn orig_cudaMallocHost = NULL;

#ifdef __cplusplus
}
#endif
