#include <flash.h>

#include <vector>

#include "xpu_timer/common/macro.h"

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
std::vector<uint64_t> parse_fwd_shape(void **args) {
  Flash_fwd_params *params = (Flash_fwd_params *)(args[0]);
  uint64_t batch = params->b;
  uint64_t dim = params->d;
  uint64_t seqlen_q = params->seqlen_q;
  uint64_t seqlen_k = params->seqlen_k;
  uint64_t num_splits = params->num_splits;
  uint64_t head = params->h;
  return {batch, seqlen_q, seqlen_k, dim * head, num_splits};
}

EXPOSE_API
std::vector<uint64_t> parse_bwd_shape(void **args) {
  Flash_bwd_params *params = (Flash_bwd_params *)(args[0]);
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
