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

#include <VERSION_TAG/nccl.h>
#include <VERSION_TAG/src/include/comm.h>

extern "C" {

#define GET_COMM_INFO(FIELD, TYPE)                                             \
  __attribute__((visibility("default"))) TYPE get_Comm_##FIELD##_FUNCTION_TAG( \
      ncclComm_t comm) {                                                       \
    return comm->FIELD;                                                        \
  }                                                                            \
  __asm__(".symver get_Comm_" #FIELD "_FUNCTION_TAG,get_Comm_" #FIELD          \
          "@VERSION_TAG");

GET_COMM_INFO(commHash, uint64_t)
GET_COMM_INFO(rank, int)
GET_COMM_INFO(nRanks, int)
GET_COMM_INFO(nNodes, int)

#undef GET_COMM_INFO

// We need the address of comm->devComm to use as the key, so we save
// &comm->devComm.
__attribute__((visibility("default"))) void* get_Comm_devComm_FUNCTION_TAG(
    ncclComm_t comm) {
  return &comm->devComm;
}
__asm__(".symver get_Comm_devComm_FUNCTION_TAG,get_Comm_devComm@VERSION_TAG");
}
