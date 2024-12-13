#pragma once
#include <stdint.h>
#ifndef PY_TRACING_BUFFER_SIZE
#define PY_TRACING_BUFFER_SIZE 512
#define PY_TRACING_MAX_THREADS 256
#endif
#define PY_TRACING_READY_POOL 0
#define PY_TRACING_EMPTY_POOL 1
#define PY_TRACING_GC 0
#define PY_TORCH_DATALOADER 1

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PAYLOAD_UNINITIALIZED = 0,
  PAYLOAD_GC = 1,
} PayloadType;

typedef union {
  int gc_debug[2];
} Payload;

// debug data should be small, maybe use void* to hold memory for more complex
// struct
typedef struct {
  uint64_t start;
  uint64_t end;
  uint32_t count;
  Payload payload;
  PayloadType type;
} XpuTimerPyTracingData;

typedef struct {
  XpuTimerPyTracingData data[PY_TRACING_BUFFER_SIZE];
  uint64_t cur;
} XpuTimerPyTracingDataArray;

#ifdef __cplusplus
}
#endif
