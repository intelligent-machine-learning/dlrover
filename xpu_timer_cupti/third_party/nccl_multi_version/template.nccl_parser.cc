#define XCCL_TRACE_ID 1
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
