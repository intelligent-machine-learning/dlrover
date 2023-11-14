// Copyright 2023 The TF-plus Authors. All Rights Reserved.
// #pragma once
#ifndef TFPLUS_FLASH_ATTN_KERNELS_FLASH_ATTENTION_H_
#define TFPLUS_FLASH_ATTN_KERNELS_FLASH_ATTENTION_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/platform/stream_executor.h"
// #include "tfplus/flash_attn/fmha.h"
#include "fmha.h"

namespace tfplus {
using namespace tensorflow;  // NOLINT(build/namespaces)

template <typename T>
void set_params_fprop(FMHA_fprop_params& params,  // NOLINT
                                                  // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t h, const size_t d,
                      // device pointers
                      const Tensor* q, const Tensor* k, const Tensor* v,
                      Tensor* out, void* cu_seqlens_q_d, void* cu_seqlens_k_d,
                      void* o_tmp_d, void* s_d, void* softmax_lse_d,
                      float p_dropout, float softmax_scale, bool is_causal,
                      int num_splits, uint32_t q_stride0, uint32_t q_stride1) {
  // default use fp16
  // Data_type data_type = DATA_TYPE_FP16;
  // Reset the parameters
  // Data_type acc_type = DT_FLOAT;
  // DATA_TYPE_FP16 : DATA_TYPE_BF16
  Data_type data_type = q->dtype() == DT_HALF ? DATA_TYPE_FP16 : DATA_TYPE_BF16;
  memset(&params, 0, sizeof(params));

  params.is_bf16 = false;
  if (data_type == DATA_TYPE_BF16) params.is_bf16 = true;

  // Set the pointers and strides.
  params.q_ptr = reinterpret_cast<void *>(
      const_cast<T *>(q->tensor<T, 3>().data()));
  params.k_ptr = reinterpret_cast<void *>(
      const_cast<T *>(k->tensor<T, 3>().data()));
  params.v_ptr = reinterpret_cast<void *>(
      const_cast<T *>(v->tensor<T, 3>().data()));
  params.q_row_stride_in_elts = q_stride0;
  params.k_row_stride_in_elts = q_stride0;
  params.v_row_stride_in_elts = q_stride0;
  params.q_head_stride_in_elts = q_stride1;
  params.k_head_stride_in_elts = q_stride1;
  params.v_head_stride_in_elts = q_stride1;
  params.o_ptr = reinterpret_cast<void*>(out->tensor<T, 3>().data());
  params.o_row_stride_in_elts = q_stride0;
  params.o_head_stride_in_elts = q_stride1;
  params.o_tmp_ptr = o_tmp_d;
  params.o_tmp_row_stride_in_elts = h * d;
  params.o_tmp_head_stride_in_elts = d;

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

  // S = softmax(P)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = d;

  // Set the different scale values.
  // const float scale_bmm1 = 1.f / sqrtf(d);
  const float scale_bmm1 = softmax_scale;

  params.scale_bmm1f = scale_bmm1;
  set_alpha(params.scale_bmm1, scale_bmm1, data_type);

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of <
  params.p_dropout_in_uint =
      uint32_t(std::floor(params.p_dropout * 4294967295.0));
  params.p_dropout_in_uint16_t =
      uint16_t(std::floor(params.p_dropout * 65535.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
  CHECK(p_dropout < 1.f);
  set_alpha(params.scale_dropout, params.rp_dropout, data_type);

  params.is_causal = is_causal;
  params.num_splits = num_splits;
}

template <typename T>
void set_params_dgrad(FMHA_dgrad_params& params,  // NOLINT
                                                  // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t h, const size_t d,
                      // device pointers
                      const Tensor* q, const Tensor* k, const Tensor* v,
                      Tensor* out, Tensor* dq, Tensor* dk, Tensor* dv,
                      void* cu_seqlens_q_d, void* cu_seqlens_k_d,
                      void* dq_tmp_d, void* do_packed_d, void* softmax_lse_d,
                      void* dsoftmax_sum_d, float p_dropout,
                      float softmax_scale, bool is_causal, int num_splits,
                      uint32_t q_stride0, uint32_t q_stride1) {
  tfplus::set_params_fprop<T>(
      params, b, seqlen_q, seqlen_k, h, d, q, k, v, out, cu_seqlens_q_d,
      cu_seqlens_k_d,
      dq_tmp_d,  // Reusing the o_tmp_ptr variable to store dq_tmp
      nullptr, softmax_lse_d, p_dropout, softmax_scale, is_causal, num_splits,
      q_stride0, q_stride1);

  // Set the pointers and strides.
  params.dq_ptr = reinterpret_cast<void*>(dq->tensor<T, 3>().data());
  params.dk_ptr = reinterpret_cast<void*>(dk->tensor<T, 3>().data());
  params.dv_ptr = reinterpret_cast<void*>(dv->tensor<T, 3>().data());
  params.dq_row_stride_in_elts = q_stride0;
  params.dk_row_stride_in_elts = q_stride0;
  params.dv_row_stride_in_elts = q_stride0;
  params.dq_head_stride_in_elts = q_stride1;
  params.dk_head_stride_in_elts = q_stride1;
  params.dv_head_stride_in_elts = q_stride1;
  params.do_ptr = do_packed_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;
}

inline void run_fmha_fwd(
    Launch_params<FMHA_fprop_params>& launch_params) {  // NOLINT
  if (launch_params.params.d <= 32) {
    run_fmha_fwd_hdim32(launch_params);
  } else if (launch_params.params.d <= 64) {
    run_fmha_fwd_hdim64(launch_params);
  } else if (launch_params.params.d <= 128) {
    run_fmha_fwd_hdim128(launch_params);
  }
}

inline void run_fmha_bwd(FMHA_dgrad_params& params, // NOLINT
                         cudaStream_t stream,
                         const bool configure,
                         cudaDeviceProp& dprops) {  // NOLINT
  if (params.d <= 32) {
    run_fmha_bwd_hdim32(params, stream, configure);
  } else if (params.d <= 64) {
    run_fmha_bwd_hdim64(params, stream, configure, dprops);
  } else if (params.d <= 128) {
    run_fmha_bwd_hdim128(params, stream, configure);
  }
}

}  // namespace tfplus
#endif  // TFPLUS_FLASH_ATTN_KERNELS_FLASH_ATTENTION_H_
