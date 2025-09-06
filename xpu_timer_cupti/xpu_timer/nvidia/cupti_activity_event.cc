#if defined(NVIDIA_WITH_CUPTI)
#include "xpu_timer/nvidia/cupti_activity_event.h"

#include <chrono>
#include <ctime>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/nvidia/nvidia_timer.h"

namespace xpu_timer {
namespace nvidia {

CuptiActivityEvent::CuptiActivityEvent()
    : GlobalCuptiExternalId(0), _bufferPool(), cupti_kernel_completed_queue() {
  CuptiActivityEvent::instance_ = this;
}

CuptiActivityEvent* CuptiActivityEvent::getInstance() {
  std::call_once(init_flag_, []() { new CuptiActivityEvent(); });
  return instance_;
}

CuptiActivityEvent::~CuptiActivityEvent() {
  CuptiActivityEvent::instance_ = nullptr;
}

void CuptiActivityEvent::initCuptiActivity() {
  CUPTI_CALL(cuptiActivityRegisterTimestampCallback([]() -> uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }));

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequestedTrampoline,
                                            bufferCompletedTrampoline));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  // Enable CUDA runtime and driver activity kinds for CUPTI to provide
  // correlation ids.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));

  size_t attrValue = 1, attrValueSize = sizeof(size_t);
  CUPTI_CALL(
      cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
                                &attrValueSize, &attrValue));

  memset(CorrelationId2ExternalId, -1,
         MAX_CORRELATION2EXTERANL_ID_SIZE * sizeof(uint32_t));
}

void CuptiActivityEvent::closeCuptiActivity() {
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  // Disable CUDA runtime and driver activity kinds for CUPTI to provide
  // correlation ids.
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));

  CUPTI_CALL(cuptiFinalize());
}

void CUPTIAPI CuptiActivityEvent::bufferRequestedTrampoline(
    uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  CuptiActivityEvent::instance_->bufferRequested(buffer, size, maxNumRecords);
}

void CuptiActivityEvent::bufferRequested(uint8_t** buffer, size_t* size,
                                         size_t* maxNumRecords) {
  *buffer = _bufferPool.getObject();
  *size = _bufferPool.getElementSize();
  memset(*buffer, 0, *size);
  *maxNumRecords = 0;
}

void CUPTIAPI CuptiActivityEvent::bufferCompletedTrampoline(CUcontext ctx,
                                                            uint32_t streamId,
                                                            uint8_t* buffer,
                                                            size_t size,
                                                            size_t validSize) {
  CuptiActivityEvent::instance_->bufferCompleted(ctx, streamId, buffer, size,
                                                 validSize);
}

void CuptiActivityEvent::CUPTIAPI bufferCompleted(CUcontext ctx,
                                                  uint32_t streamId,
                                                  uint8_t* buffer, size_t size,
                                                  size_t validSize) {
  CUpti_Activity* record = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
          CUpti_ActivityExternalCorrelation* pExternalCorrelationRecord =
              (CUpti_ActivityExternalCorrelation*)record;
          if (pExternalCorrelationRecord->externalKind ==
              CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2) {
            uint32_t offset = pExternalCorrelationRecord->correlationId %
                              MAX_CORRELATION2EXTERANL_ID_SIZE;
            CorrelationId2ExternalId[offset] =
                pExternalCorrelationRecord->externalId;
#if 0
            XLOG(INFO) << "externalkernel  coll: " << offset
                       << " exid: " << pExternalCorrelationRecord->externalId;
#endif
          }
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          setActivityKernel4(record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          setActivitymemcpy5(record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
          const auto* kernel = (const CUpti_ActivityAPI*)record;
          uint32_t external_id =
              CorrelationId2ExternalId[kernel->correlationId %
                                       MAX_CORRELATION2EXTERANL_ID_SIZE];
#if 0
          XLOG(INFO) << "externalkernel  coll: "
                     << kernel->correlationId % MAX_CORRELATION2EXTERANL_ID_SIZE
                     << " runexid: " << external_id;
#endif
          if (external_id != -1) {
            CorrelationId2ExternalId[kernel->correlationId %
                                     MAX_CORRELATION2EXTERANL_ID_SIZE] = -1;
          }
          break;
        }
        default:
          break;
      }
    }
  } while (status == CUPTI_SUCCESS);

  int bufferPoolSize;
  _bufferPool.returnObject(buffer, &bufferPoolSize);
}

uint32_t CuptiActivityEvent::pushCuptiExternalActivity() {
  uint32_t external_id;
  {
    std::lock_guard<std::mutex> lock(GlobalCuptiExternalIdMtx);
    external_id = GlobalCuptiExternalId;
    GlobalCuptiExternalId += 1;
    GlobalCuptiExternalId %= MAX_EXTERNAL_ID2CUPTI_INFO_SIZE;
  }
  cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2, external_id);
  GlobalExternalId2CuptiInfo[external_id].start_timestamp_ = 0;
  return external_id;
}

void CuptiActivityEvent::popCuptiExternalActivity() {
  uint64_t id;
  cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
                                        &id);
}

bool CuptiActivityEvent::getCuptiInfoByExternalId(
    uint32_t cupti_external_id, nvidia::NvidiaLaunchInfo* cupti_info) {
  bool ready =
      GlobalExternalId2CuptiInfo[cupti_external_id].start_timestamp_ != 0;
  if (!ready) {
    return ready;
  }
  *cupti_info = GlobalExternalId2CuptiInfo[cupti_external_id];
  GlobalExternalId2CuptiInfo[cupti_external_id].start_timestamp_ = 0;
  return ready;
}
void CuptiActivityEvent::setActivityKernel4(CUpti_Activity* record) {
  const auto* kernel = (const CUpti_ActivityKernel4*)record;
  uint32_t external_id =
      CorrelationId2ExternalId[kernel->correlationId %
                               MAX_CORRELATION2EXTERANL_ID_SIZE];
#if 0
  XLOG(INFO) << "setActivityKernel4: " << external_id << " " << kernel->name;
#endif

  if (external_id == -1) {
    return;
  }

  nvidia::NvidiaLaunchInfo* cupti_info =
      &GlobalExternalId2CuptiInfo[external_id];

  cupti_info->start_timestamp_ = kernel->start / 1000;
  cupti_info->duration_ = (kernel->end - kernel->start) / 1000;
  cupti_info->duration_ =
      cupti_info->duration_ == 0 ? 1 : cupti_info->duration_;

  cupti_info->grid_dim_.x = kernel->gridX;
  cupti_info->grid_dim_.y = kernel->gridY;
  cupti_info->grid_dim_.z = kernel->gridZ;

  cupti_info->block_dim_.x = kernel->blockX;
  cupti_info->block_dim_.y = kernel->blockY;
  cupti_info->block_dim_.z = kernel->blockZ;

  cupti_info->stream_id_ = kernel->streamId;

  cupti_kernel_completed_queue.push(new uint32_t(external_id));

  CorrelationId2ExternalId[kernel->correlationId %
                           MAX_CORRELATION2EXTERANL_ID_SIZE] = -1;
}
void CuptiActivityEvent::setActivitymemcpy5(CUpti_Activity* record) {
  const auto* kernel = (const CUpti_ActivityMemcpy5*)record;
  uint32_t external_id =
      CorrelationId2ExternalId[kernel->correlationId %
                               MAX_CORRELATION2EXTERANL_ID_SIZE];
#if 0
  XLOG(INFO) << "setActivitymemcpy5: " << external_id;
#endif

  if (external_id == -1) {
    return;
  }

  nvidia::NvidiaLaunchInfo* cupti_info =
      &GlobalExternalId2CuptiInfo[external_id];

  cupti_info->start_timestamp_ = kernel->start / 1000;
  cupti_info->duration_ = (kernel->end - kernel->start) / 1000;
  cupti_info->duration_ =
      cupti_info->duration_ == 0 ? 1 : cupti_info->duration_;

  cupti_info->stream_id_ = kernel->streamId;
  cupti_kernel_completed_queue.push(new uint32_t(external_id));

  CorrelationId2ExternalId[kernel->correlationId %
                           MAX_CORRELATION2EXTERANL_ID_SIZE] = -1;
}
uint32_t CuptiActivityEvent::getNextCompletedExternalId() {
  uint32_t* external_id = cupti_kernel_completed_queue.front();
  return external_id == nullptr ? -1 : *external_id;
}
void CuptiActivityEvent::popNextCompletedExternalId() {
  cupti_kernel_completed_queue.pop_front();
}
bool CuptiActivityEvent::hasNextCompletedExternalId() {
  return cupti_kernel_completed_queue.front() != nullptr;
}

}  // namespace nvidia

}  // namespace xpu_timer
#endif
