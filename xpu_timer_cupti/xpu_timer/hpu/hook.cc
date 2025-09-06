#include "xpu_timer/hpu/hook.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/manager.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/hpu/hpu_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (*dlsymFn)(void* handle, const char* name);

void* dlsym(void* handle, const char* name) {
  static dlsymFn real_dlsym = NULL;
  static bool hook_matmul = xpu_timer::util::EnvVarRegistry::GetEnvVar<bool>(
      "XPU_TIMER_HPU_HOOK_MATMUL");
  if (real_dlsym == NULL) {
    // dlvsym(), provided by glibc since version 2.1, does the same as dlsym()
    // but takes a version string as an additional argument.
    // the version is from `readelf -a -W libdl.so | grep dlsym`
    real_dlsym = (dlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    // To get the newest version, as well as a potentially one of another
    // interceptor in the same process, to do an unversioned query again:
    // https://stackoverflow.com/questions/15599026/how-can-i-intercept-dlsym-calls-using-ld-preload
    real_dlsym = (dlsymFn)real_dlsym(RTLD_NEXT, "dlsym");
  }
  if (hook_matmul) {
    if (!strcmp(name, "aclnnMatmulGetWorkspaceSize")) {
      SETUP_DLSYM_WITH_HPU_OPAPI(aclnnMatmulGetWorkspaceSize);
      return (void*)aclnnMatmulGetWorkspaceSize;
    } else if (!strcmp(name, "aclnnMatmul")) {
      SETUP_DLSYM_WITH_HPU_OPAPI(aclnnMatmul);
      return (void*)aclnnMatmul;
    } else if (!(strcmp(name, "aclnnGroupedMatmulV2GetWorkspaceSize"))) {
      SETUP_DLSYM_WITH_HPU_OPAPI(aclnnGroupedMatmulV2GetWorkspaceSize);
      return (void*)aclnnGroupedMatmulV2GetWorkspaceSize;
    } else if (!strcmp(name, "aclnnGroupedMatmulV2")) {
      SETUP_DLSYM_WITH_HPU_OPAPI(aclnnGroupedMatmulV2);
      return (void*)aclnnGroupedMatmulV2;
    }
  }
  if (!strcmp(name, "HcclAlltoAllV")) {
    SETUP_DLSYM_WITH_HPU_HCCL_REAL(HcclAlltoAllV);
    return (void*)HcclAlltoAllV;
  } else if (!strcmp(name, "HcclReduce")) {
    SETUP_DLSYM_WITH_HPU_HCCL_REAL(HcclReduce);
    return (void*)HcclReduce;
  } else if (!strcmp(name, "HcclScatter")) {
    SETUP_DLSYM_WITH_HPU_HCCL_REAL(HcclScatter);
    return (void*)HcclScatter;
  } else if (!strcmp(name, "HcclBatchSendRecv")) {
    SETUP_DLSYM_WITH_HPU_HCCL_REAL(HcclBatchSendRecv);
    return (void*)HcclBatchSendRecv;
  } else if (!strcmp(name, "HcclAlltoAll")) {
    SETUP_DLSYM_WITH_HPU_HCCL_REAL(HcclAlltoAll);
    return (void*)HcclAlltoAll;
  }
  return real_dlsym(handle, name);
}

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor* self,
                                        const aclTensor* other, aclTensor* out,
                                        int8_t cubeMathType,
                                        uint64_t* workspace_size,
                                        aclOpExecutor** executor) {
  aclnnStatus ret_status = orig_aclnnMatmulGetWorkspaceSize(
      self, other, out, cubeMathType, workspace_size, executor);

  if (!::xpu_timer::util::config::GlobalConfig::enable) {
    return ret_status;
  }
  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .intercept_manager.interceptMatmulInfo(self, other, executor);
  return ret_status;
}

aclnnStatus aclnnMatmul(void* workspace, uint64_t workspaceSize,
                        aclOpExecutor* executor, aclrtStream stream) {
  if (!::xpu_timer::util::config::GlobalConfig::enable) {
    return orig_aclnnMatmul(workspace, workspaceSize, executor, stream);
  }

  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleMatmul(executor);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MatmulMetrics::TYPE);

  auto ret_status =
      orig_aclnnMatmul(workspace, workspaceSize, executor, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);
  return ret_status;
}

// aclnnGroupedMatmulV2
aclnnStatus aclnnGroupedMatmulV2GetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
    const aclTensorList* offsetOptional,
    const aclTensorList* antiquantScaleOptional,
    const aclTensorList* antiquantOffsetOptional,
    const aclIntArray* groupListOptional, int64_t splitItem, int64_t groupType,
    const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor) {
  aclnnStatus ret_status = orig_aclnnGroupedMatmulV2GetWorkspaceSize(
      x, weight, biasOptional, scaleOptional, offsetOptional,
      antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
      splitItem, groupType, y, workspaceSize, executor);
  if (!::xpu_timer::util::config::GlobalConfig::enable) {
    return ret_status;
  }

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .intercept_manager.interceptGroupedMatmulV2Info(
          x, weight, groupListOptional, splitItem, groupType, executor);
  return ret_status;
}

aclnnStatus aclnnGroupedMatmulV2(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream) {
  if (!::xpu_timer::util::config::GlobalConfig::enable) {
    return orig_aclnnGroupedMatmulV2(workspace, workspaceSize, executor,
                                     stream);
  }

  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleGroupedMatmulV2(executor);
  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MatmulMetrics::TYPE);

  auto ret_status =
      orig_aclnnGroupedMatmulV2(workspace, workspaceSize, executor, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);
  return ret_status;
}

HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count,
                         HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                         aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclAllReduce);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm,
                              stream);
  }

  std::string func_name = "HcclAllReduce";
  std::string coll_type = "AllReduce";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(count, dataType, comm, func_name,
                                              coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult =
      orig_HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}
HcclResult HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType,
                         uint32_t root, HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclBroadcast);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclBroadcast(buf, count, dataType, root, comm, stream);
  }

  std::string func_name = "HcclBroadcast";
  std::string coll_type = "Broadcast";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(count, dataType, comm, func_name,
                                              coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclBroadcast(buf, count, dataType, root, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclReduceScatter(void* sendBuf, void* recvBuf, uint64_t recvCount,
                             HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclReduceScatter);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op,
                                  comm, stream);
  }

  std::string func_name = "HcclReduceScatter";
  std::string coll_type = "ReduceScatter";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(recvCount, dataType, comm,
                                              func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType,
                                          op, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclReduce(void* sendBuf, void* recvBuf, uint64_t count,
                      HcclDataType dataType, HcclReduceOp op, uint32_t root,
                      HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclReduce);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclReduce(sendBuf, recvBuf, count, dataType, op, root, comm,
                           stream);
  }

  std::string func_name = "HcclReduce";
  std::string coll_type = "Reduce";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(count, dataType, comm, func_name,
                                              coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclReduce(sendBuf, recvBuf, count, dataType, op, root,
                                   comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclAlltoAll(const void* sendBuf, uint64_t sendCount,
                        HcclDataType sendType, const void* recvBuf,
                        uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclAlltoAll);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount,
                             recvType, comm, stream);
  }

  std::string func_name = "HcclAlltoAll";
  std::string coll_type = "AlltoAll";
  // TODO(jingjun): calculate all to all
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(recvCount, recvType, comm,
                                              func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf,
                                     recvCount, recvType, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclAlltoAllV(const void* sendBuf, const void* sendCounts,
                         const void* sdispls, HcclDataType sendType,
                         const void* recvBuf, const void* recvCounts,
                         const void* rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclAlltoAllV);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclAlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf,
                              recvCounts, rdispls, recvType, comm, stream);
  }

  std::string func_name = "HcclAlltoAllV";
  std::string coll_type = "AlltoAllV";
  // TODO(jingjun): calculate all to all v
  auto fn =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .intercept_manager.handleHccl(*((uint64_t*)recvCounts), recvType,
                                        comm, func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult =
      orig_HcclAlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf,
                         recvCounts, rdispls, recvType, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount,
                         HcclDataType dataType, HcclComm comm,
                         aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclAllGather);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm,
                              stream);
  }

  std::string func_name = "HcclAllGather";
  std::string coll_type = "AllGather";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(sendCount, dataType, comm,
                                              func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  HcclResult retResult =
      orig_HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclBarrier);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclBarrier(comm, stream);
  }

  std::string func_name = "HcclBarrier";
  std::string coll_type = "Barrier";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(1, HCCL_DATA_TYPE_FP32, comm,
                                              func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclBarrier(comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclScatter(void* sendBuf, void* recvBuf, uint64_t recvCount,
                       HcclDataType dataType, uint32_t root, HcclComm comm,
                       aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclScatter);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm,
                            stream);
  }

  std::string func_name = "HcclScatter";
  std::string coll_type = "Scatter";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(recvCount, dataType, comm,
                                              func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclScatter(sendBuf, recvBuf, recvCount, dataType, root,
                                    comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType,
                    uint32_t destRank, HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclSend);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclSend(sendBuf, count, dataType, destRank, comm, stream);
  }

  std::string func_name = "HcclSend";
  std::string coll_type = "Send";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(count, dataType, comm, func_name,
                                              coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult =
      orig_HcclSend(sendBuf, count, dataType, destRank, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}
HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType,
                    uint32_t srcRank, HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclRecv);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclRecv(recvBuf, count, dataType, srcRank, comm, stream);
  }

  std::string func_name = "HcclRecv";
  std::string coll_type = "Recv";
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(count, dataType, comm, func_name,
                                              coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult =
      orig_HcclRecv(recvBuf, count, dataType, srcRank, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum,
                             HcclComm comm, aclrtStream stream) {
  SETUP_DLSYM_WITH_HPU_HCCL(HcclBatchSendRecv);

  if (!::xpu_timer::util::config::GlobalConfig::enable || isNranksLE1(comm)) {
    return orig_HcclBatchSendRecv(sendRecvInfo, itemNum, comm, stream);
  }

  std::string func_name = "HcclBatchSendRecv";
  std::string coll_type = "BatchSendRecv";
  // TODO(jingjun): support batched send recv
  auto fn = xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
                .intercept_manager.handleHccl(itemNum, HCCL_DATA_TYPE_FP32,
                                              comm, func_name, coll_type);

  auto event =
      xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
          .getEvent();

  event->reset(stream, fn, xpu_timer::constant::Metrics::CollMetrics::TYPE);

  auto retResult = orig_HcclBatchSendRecv(sendRecvInfo, itemNum, comm, stream);

  xpu_timer::GpuTimerManager<xpu_timer::hpu::HpuTimer>::getInstance()
      .recordEvent(event);

  return retResult;
}

bool isNranksLE1(HcclComm comm) {
  uint32_t nranks;
  HcclGetRankSize(comm, &nranks);
  return nranks <= 1;
}
#ifdef __cplusplus
}
#endif
