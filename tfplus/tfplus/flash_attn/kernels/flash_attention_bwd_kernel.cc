// Copyright 2023 The TF-plus Authors. All Rights Reserved.
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include <iostream>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tfplus/flash_attn/kernels/flash_attention.h"
// #include "tfplus/flash_attn/kernels/fmha.h"
#include "fmha.h"

namespace tfplus {
using namespace tensorflow;  // NOLINT(build/namespaces)

using GPUDevice = Eigen::GpuDevice;

template <typename T>
class FMHABackwardOp : public OpKernel {
 public:
  explicit FMHABackwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Get the attributes
    OP_REQUIRES_OK(ctx, ctx->GetAttr("p_dropout", &p_dropout_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("softmax_scale", &softmax_scale_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_tensors", &zero_tensors_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_causal", &is_causal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("return_softmax", &return_softmax_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    is_dropout_ = p_dropout_ > 0.0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaGetDeviceProperties(&dprops_, current_device);
    CUDAPhiloxRandomGenerator gen();
  }

  ~FMHABackwardOp() override {}

  void Compute(OpKernelContext* ctx) override {
    // Get the input tensors
    const Tensor& query_tensor = ctx->input(0);
    // VLOG(0) << "query_tensor: " << query_tensor.DebugString();
    const Tensor& key_tensor = ctx->input(1);
    const Tensor& value_tensor = ctx->input(2);
    const Tensor& cu_seqlens_q = ctx->input(3);
    const Tensor& cu_seqlens_k = ctx->input(4);
    const Tensor& output_tensor = ctx->input(5);
    const Tensor& gradient_tensor = ctx->input(6);
    const Tensor& softmax_lse_tensor = ctx->input(7);
    const Tensor& max_seqlen_q_tensor = ctx->input(8);
    const Tensor& max_seqlen_k_tensor = ctx->input(9);
    const Tensor& rng_state_tensor = ctx->input(10);
    int32 max_seqlen_q_ =
        *max_seqlen_q_tensor.flat<int32>().data();
    int32 max_seqlen_k_ =
        *max_seqlen_k_tensor.flat<int32>().data();
    // Check the shapes of the input tensors
    Tensor* dq = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query_tensor.shape(), &dq));
    Tensor* dk = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, key_tensor.shape(), &dk));
    Tensor* dv = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, key_tensor.shape(), &dv));
    auto launch = &tfplus::run_fmha_bwd;

    int major = dprops_.major;
    int minor = dprops_.minor;
    if (major == 7 && minor == 5) is_sm75_ = true;
    int batch_size = cu_seqlens_q.NumElements() - 1;
    const int total_q = query_tensor.dim_size(0);
    const int num_heads = query_tensor.dim_size(1);
    const int head_size = query_tensor.dim_size(2);
    const int total_k = key_tensor.dim_size(0);
    int blocksize_c =
        (head_size > 64 || (is_sm75_ && head_size > 32)) ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int max_seqlen_k =
        ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if (max_seqlen_k_ <= 128) {
      max_seqlen_k = 128;
    } else if (max_seqlen_k_ <= 256) {
      max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > blocksize_c;
    Tensor dq_tmp;
    if (loop) {
      ctx->allocate_temp(DT_FLOAT, {total_q, num_heads, head_size}, &dq_tmp);
    }
    Tensor softmax_d;
    ctx->allocate_temp(DT_FLOAT, {batch_size, num_heads, max_seqlen_q},
                       &softmax_d);
    FMHA_dgrad_params params;
    tfplus::set_params_dgrad<T>(
        params, batch_size, max_seqlen_q, max_seqlen_k, num_heads, head_size,
        &query_tensor, &key_tensor, &value_tensor,
        const_cast<Tensor *>(&output_tensor), dq, dk, dv,
        reinterpret_cast<void *>(
            const_cast<int32 *>(cu_seqlens_q.tensor<int32, 1>().data())),
        reinterpret_cast<void *>(
            const_cast<int32 *>(cu_seqlens_k.tensor<int32, 1>().data())),
        loop ? reinterpret_cast<void *>(dq_tmp.tensor<float, 3>().data())
             : nullptr,
        reinterpret_cast<void *>(
            const_cast<T *>(gradient_tensor.tensor<T, 3>().data())),
        reinterpret_cast<void *>(
            const_cast<float *>(softmax_lse_tensor.tensor<float, 3>().data())),
        reinterpret_cast<void *>(softmax_d.tensor<float, 3>().data()),
        p_dropout_, softmax_scale_, is_causal_, num_splits_,
        num_heads * head_size, head_size);
    auto stream = GetGpuStream(ctx);
    launch(params, stream, /*configure=*/true, dprops_);

    // number of times random will be generated per thread, to offset philox
    // counter in thc random state We use a custom RNG that increases the offset
    // by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    if (is_dropout_) {
      params.rng_state = reinterpret_cast<uint64_t*>(
          const_cast<uint64*>(rng_state_tensor.tensor<uint64, 1>().data()));
    }
    launch(params, stream, /*configure=*/false, dprops_);
  }

 private:
  // int max_seqlen_k_;
  // int max_seqlen_q_;
  float p_dropout_;
  float softmax_scale_;
  bool zero_tensors_;
  bool is_causal_;
  bool is_dropout_;
  bool return_softmax_;
  int num_splits_;
  bool is_sm75_;
  cudaDeviceProp dprops_;
  CUDAPhiloxRandomGenerator gen;
  DataType dtype_;
};

// Register the GPU kernels.

#define REGISTER_FLASH_ATTENTION(type)                        \
  REGISTER_KERNEL_BUILDER(Name("FMHABackward")                \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("max_seqlen_q")     \
                              .HostMemory("max_seqlen_k")     \
                              .TypeConstraint<type>("dtype"), \
                          FMHABackwardOp<type>);

REGISTER_FLASH_ATTENTION(Eigen::half)
REGISTER_FLASH_ATTENTION(bfloat16)
#undef REGISTER_FLASH_ATTENTION

}  // namespace tfplus

#endif  // GOOGLE_CUDA
