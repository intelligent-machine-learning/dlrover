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

#include <flash.h>

#include <vector>

#include "xpu_timer/common/macro.h"

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
std::vector<uint64_t> parse_fwd_shape(void** args) {
  Flash_fwd_params* params = (Flash_fwd_params*)(args[0]);
  uint64_t batch = params->b;
  uint64_t dim = params->d;
  uint64_t seqlen_q = params->seqlen_q;
  uint64_t seqlen_k = params->seqlen_k;
  uint64_t num_splits = params->num_splits;
  uint64_t head = params->h;
  return {batch, seqlen_q, seqlen_k, dim * head, num_splits};
}

EXPOSE_API
std::vector<uint64_t> parse_bwd_shape(void** args) {
  Flash_bwd_params* params = (Flash_bwd_params*)(args[0]);
  uint64_t batch = params->b;
  uint64_t dim = params->d;
  uint64_t seqlen_q = params->seqlen_q;
  uint64_t seqlen_k = params->seqlen_k;
  uint64_t num_splits = params->num_splits;
  uint64_t head = params->h;
  return {batch, seqlen_q, seqlen_k, dim * head, num_splits};
}
#ifdef __cplusplus
}
#endif
