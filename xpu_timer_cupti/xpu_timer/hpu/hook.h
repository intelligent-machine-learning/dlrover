#pragma once
#include <dlfcn.h>

#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// dlsym
EXPOSE_API
void* dlsym(void* handle, const char* name);

// aclnnMatmul
typedef aclnnStatus (*aclnnMatmulFn)(void*, uint64_t, aclOpExecutor*,
                                     aclrtStream);

static aclnnMatmulFn orig_aclnnMatmul = NULL;

typedef aclnnStatus (*aclnnMatmulGetWorkspaceSizeFn)(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspace_size, aclOpExecutor** executor);

static aclnnMatmulGetWorkspaceSizeFn orig_aclnnMatmulGetWorkspaceSize = NULL;

EXPOSE_API aclnnStatus aclnnMatmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspace_size, aclOpExecutor** executor);

EXPOSE_API aclnnStatus aclnnMatmul(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);
// aclnnGroupedMatmulV2
typedef aclnnStatus (*aclnnGroupedMatmulV2GetWorkspaceSizeFn)(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
    const aclTensorList* offsetOptional,
    const aclTensorList* antiquantScaleOptional,
    const aclTensorList* antiquantOffsetOptional,
    const aclIntArray* groupListOptional, int64_t splitItem, int64_t groupType,
    const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);

typedef aclnnStatus (*aclnnGroupedMatmulV2Fn)(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream);

static aclnnGroupedMatmulV2GetWorkspaceSizeFn
    orig_aclnnGroupedMatmulV2GetWorkspaceSize = NULL;
static aclnnGroupedMatmulV2Fn orig_aclnnGroupedMatmulV2 = NULL;

EXPOSE_API aclnnStatus aclnnGroupedMatmulV2GetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
    const aclTensorList* offsetOptional,
    const aclTensorList* antiquantScaleOptional,
    const aclTensorList* antiquantOffsetOptional,
    const aclIntArray* groupListOptional, int64_t splitItem, int64_t groupType,
    const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);
EXPOSE_API aclnnStatus aclnnGroupedMatmulV2(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

// HcclAllReduce
typedef HcclResult (*HcclAllReduceFn)(void* sendBuf, void* recvBuf,
                                      uint64_t count, HcclDataType dataType,
                                      HcclReduceOp op, HcclComm comm,
                                      aclrtStream stream);
static HcclAllReduceFn orig_HcclAllReduce = NULL;

EXPOSE_API
HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count,
                         HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                         aclrtStream stream);

// HcclBroadcast
typedef HcclResult (*HcclBroadcastFn)(void* buf, uint64_t count,
                                      HcclDataType dataType, uint32_t root,
                                      HcclComm comm, aclrtStream stream);
static HcclBroadcastFn orig_HcclBroadcast = NULL;
EXPOSE_API
HcclResult HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType,
                         uint32_t root, HcclComm comm, aclrtStream stream);

// HcclReduceScatter
typedef HcclResult (*HcclReduceScatterFn)(void* sendBuf, void* recvBuf,
                                          uint64_t recvCount,
                                          HcclDataType dataType,
                                          HcclReduceOp op, HcclComm comm,
                                          aclrtStream stream);
static HcclReduceScatterFn orig_HcclReduceScatter = NULL;
EXPOSE_API
HcclResult HcclReduceScatter(void* sendBuf, void* recvBuf, uint64_t recvCount,
                             HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream);

// HcclReduce
typedef HcclResult (*HcclReduceFn)(void* sendBuf, void* recvBuf, uint64_t count,
                                   HcclDataType dataType, HcclReduceOp op,
                                   uint32_t root, HcclComm comm,
                                   aclrtStream stream);
static HcclReduceFn orig_HcclReduce = NULL;
EXPOSE_API
HcclResult HcclReduce(void* sendBuf, void* recvBuf, uint64_t count,
                      HcclDataType dataType, HcclReduceOp op, uint32_t root,
                      HcclComm comm, aclrtStream stream);

// HcclAlltoAll
typedef HcclResult (*HcclAlltoAllFn)(const void* sendBuf, uint64_t sendCount,
                                     HcclDataType sendType, const void* recvBuf,
                                     uint64_t recvCount, HcclDataType recvType,
                                     HcclComm comm, aclrtStream stream);
static HcclAlltoAllFn orig_HcclAlltoAll = NULL;
EXPOSE_API
HcclResult HcclAlltoAll(const void* sendBuf, uint64_t sendCount,
                        HcclDataType sendType, const void* recvBuf,
                        uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream);

// HcclAlltoAllV
typedef HcclResult (*HcclAlltoAllVFn)(
    const void* sendBuf, const void* sendCounts, const void* sdispls,
    HcclDataType sendType, const void* recvBuf, const void* recvCounts,
    const void* rdispls, HcclDataType recvType, HcclComm comm,
    aclrtStream stream);

static HcclAlltoAllVFn orig_HcclAlltoAllV = NULL;
EXPOSE_API
HcclResult HcclAlltoAllV(const void* sendBuf, const void* sendCounts,
                         const void* sdispls, HcclDataType sendType,
                         const void* recvBuf, const void* recvCounts,
                         const void* rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream);

// HcclAllGather
typedef HcclResult (*HcclAllGatherFn)(void*, void*, uint64_t, HcclDataType,
                                      HcclComm, aclrtStream);
static HcclAllGatherFn orig_HcclAllGather = NULL;

EXPOSE_API
HcclResult HcclAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount,
                         HcclDataType dataType, HcclComm comm,
                         aclrtStream stream);

// HcclBarrier
typedef HcclResult (*HcclBarrierFn)(HcclComm comm, aclrtStream stream);

static HcclBarrierFn orig_HcclBarrier = NULL;
EXPOSE_API
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);

// HcclScatter
typedef HcclResult (*HcclScatterFn)(void* sendBuf, void* recvBuf,
                                    uint64_t recvCount, HcclDataType dataType,
                                    uint32_t root, HcclComm comm,
                                    aclrtStream stream);
static HcclScatterFn orig_HcclScatter = NULL;
EXPOSE_API
HcclResult HcclScatter(void* sendBuf, void* recvBuf, uint64_t recvCount,
                       HcclDataType dataType, uint32_t root, HcclComm comm,
                       aclrtStream stream);

// HcclSend
typedef HcclResult (*HcclSendFn)(void* sendBuf, uint64_t count,
                                 HcclDataType dataType, uint32_t destRank,
                                 HcclComm comm, aclrtStream stream);
static HcclSendFn orig_HcclSend = NULL;
EXPOSE_API
HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType,
                    uint32_t destRank, HcclComm comm, aclrtStream stream);

// HcclRecv
typedef HcclResult (*HcclRecvFn)(void* recvBuf, uint64_t count,
                                 HcclDataType dataType, uint32_t srcRank,
                                 HcclComm comm, aclrtStream stream);
static HcclRecvFn orig_HcclRecv = NULL;
EXPOSE_API
HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType,
                    uint32_t srcRank, HcclComm comm, aclrtStream stream);

// HcclBatchSendRecv
typedef HcclResult (*HcclBatchSendRecvFn)(HcclSendRecvItem* sendRecvInfo,
                                          uint32_t itemNum, HcclComm comm,
                                          aclrtStream stream);
static HcclBatchSendRecvFn orig_HcclBatchSendRecv = NULL;
EXPOSE_API
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum,
                             HcclComm comm, aclrtStream stream);

// judge the rank of comm is <= 1
bool isNranksLE1(HcclComm comm);

#ifdef __cplusplus
}
#endif
