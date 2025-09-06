#pragma once
#include <string>

namespace xpu_timer {
namespace platform {
std::string getDeviceName();
}
}  // namespace xpu_timer

#if defined(XPU_NVIDIA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#if defined(CUDART_VERSION)

#if CUDART_VERSION >= 11000
#define CUDA_VERSION_11
#endif

#if CUDART_VERSION >= 11080
#define CUDA_VERSION_11_8
#define CUDA_LAUNCH_EXC
#define CUDA_FP8
#endif

#if CUDART_VERSION >= 12000
#define CUDA_VERSION_12
#endif

#else
#error "CUDART_VERSION is not defined; ensure CUDA runtime is included."
#endif

#elif defined(XPU_NPU)

#include <comm.h>
#include <hggcrt.h>
#include <pccl.h>

#define cudaDeviceProp hggcDeviceProp
#define cudaErrorNotReady hggcErrorNotReady
#define cudaError_t hggcError_t
#define cudaEventCreate hggcEventCreate
#define cudaEventElapsedTime hggcEventElapsedTime
#define cudaEventQuery hggcEventQuery
#define cudaEventRecord hggcEventRecord
#define cudaEventSynchronize hggcEventSynchronize
#define cudaEvent_t hggcEvent_t
#define cudaGetDeviceCount hggcGetDeviceCount
#define cudaGetDeviceProperties hggcGetDeviceProperties
#define cudaGetErrorString hggcGetErrorString
#define cudaLaunchConfig_t hggcLaunchConfig_t
#define cudaLaunchKernel hggcLaunchKernel
#define cudaLaunchKernelExC hggcLaunchKernelExC
#define cudaSetDevice hggcSetDevice
#define cudaStream_t hggcStream_t
#define cudaSuccess hggcSuccess
#define cudaMemcpyKind hggcMemcpyKind
#define cudaMemPool_t hggcMemPool_t

#define ncclComm_t pcclComm_t
#define ncclDataType_t pcclDataType_t
#define ncclResult_t pcclResult_t
#define ncclRedOp_t pcclRedOp_t
#define ncclInt8 pcclInt8
#define ncclChar pcclChar
#define ncclUint8 pcclUint8
#define ncclInt32 pcclInt32
#define ncclInt pcclInt
#define ncclUint32 pcclUint32
#define ncclInt64 pcclInt64
#define ncclUint64 pcclUint64
#define ncclFloat16 pcclFloat16
#define ncclBfloat16 pcclBfloat16
#define ncclHalf pcclHalf
#define ncclFloat32 pcclFloat32
#define ncclFloat pcclFloat
#define ncclFloat64 pcclFloat64
#define ncclDouble pcclDouble
#define ncclNumTypes pcclNumTypes

#elif defined(XPU_HPU)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_matmul.h>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#endif

#if defined(NVIDIA_WITH_CUPTI)
#include <cupti.h>
#endif
