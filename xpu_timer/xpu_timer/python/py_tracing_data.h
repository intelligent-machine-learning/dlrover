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
