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

#include "xpu_timer/python/py_tracing_data.h"

#ifdef __cplusplus
extern "C" {
#endif
// get empty gc data from pool, used in gc_callback.c
__attribute__((visibility("default"))) XpuTimerPyTracingDataArray*
xpu_timer_get_empty_py_tracing_data_array(int);
// get full gc data from pool, used in manager.cc
__attribute__((visibility("default"))) XpuTimerPyTracingDataArray*
xpu_timer_get_full_py_tracing_data_array(int);
// get partial gc data from pool, used in manager.cc
// there are only on partial gc data which is not full
__attribute__((visibility("default"))) XpuTimerPyTracingDataArray*
xpu_timer_get_partial_py_tracing_data_array(int);
// return full gc data to ready pool, used in gc_callback.c
// return empty gc data to empty pool, used in manager.cc
__attribute__((visibility("default"))) void
xpu_timer_return_py_tracing_data_array(XpuTimerPyTracingDataArray*, int type,
                                       int name);
// register callback
__attribute__((visibility("default"))) void xpu_timer_register_tracing(
    const char**, int, char**);
// switch gc recording
__attribute__((visibility("default"))) void xpu_timer_switch_py_tracing(int s);
// get gc count, maybe add latency later
__attribute__((visibility("default"))) int64_t
xpu_timer_get_py_tracing_count(int);

__attribute__((visibility("default"))) void xpu_timer_tracing_debug();
#ifdef __cplusplus
}
#endif
