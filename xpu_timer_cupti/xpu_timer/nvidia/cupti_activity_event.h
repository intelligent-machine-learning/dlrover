#pragma once
#if defined(NVIDIA_WITH_CUPTI)
#include <cuda_runtime_api.h>
#include <cupti.h>

#include <unordered_map>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/macro.h"
#include "xpu_timer/nvidia/nvidia_timer.h"

#define CUPTI_CALL(call)                      \
  do {                                        \
    CUptiResult _status = call;               \
    if (_status != CUPTI_SUCCESS) {           \
      const char* errstr;                     \
      cuptiGetResultString(_status, &errstr); \
      XLOG(ERROR) << errstr << std::endl;     \
      throw std::runtime_error(errstr);       \
    }                                         \
  } while (0)

namespace xpu_timer {

namespace nvidia {

class CuptiActivityEvent {
 public:
  static CuptiActivityEvent* getInstance();
  ~CuptiActivityEvent();

  uint32_t pushCuptiExternalActivity();
  void popCuptiExternalActivity();
  bool getCuptiInfoByExternalId(uint32_t cupti_external_id,
                                nvidia::NvidiaLaunchInfo* cupti_info);

  void initCuptiActivity();
  void closeCuptiActivity();

  uint32_t getNextCompletedExternalId();
  void popNextCompletedExternalId();
  bool hasNextCompletedExternalId();

 private:
  static const uint32_t MAX_CORRELATION2EXTERANL_ID_SIZE = 100000;
  static const uint32_t MAX_EXTERNAL_ID2CUPTI_INFO_SIZE = 100000;

  inline static CuptiActivityEvent* instance_ = nullptr;
  inline static std::once_flag init_flag_;

  xpu_timer::util::TimerPool<uint8_t, 512 * 1024> _bufferPool;
  xpu_timer::util::BlockingDeque<uint32_t> cupti_kernel_completed_queue;
  std::mutex GlobalCuptiExternalIdMtx;
  uint32_t GlobalCuptiExternalId = 0;
  uint32_t CorrelationId2ExternalId[MAX_CORRELATION2EXTERANL_ID_SIZE];
  alignas(64) nvidia::NvidiaLaunchInfo
      GlobalExternalId2CuptiInfo[MAX_EXTERNAL_ID2CUPTI_INFO_SIZE];

  CuptiActivityEvent();

  void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size,
                                size_t* maxNumRecords);
  void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                uint8_t* buffer, size_t size, size_t validSize);

  static void CUPTIAPI bufferRequestedTrampoline(uint8_t** buffer, size_t* size,
                                                 size_t* maxNumRecords);
  static void CUPTIAPI bufferCompletedTrampoline(CUcontext ctx,
                                                 uint32_t streamId,
                                                 uint8_t* buffer, size_t size,
                                                 size_t validSize);

  inline void setActivityKernel4(CUpti_Activity* record);
  inline void setActivitymemcpy5(CUpti_Activity* record);
};
}  // namespace nvidia

}  // namespace xpu_timer
#endif
