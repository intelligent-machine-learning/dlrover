// Modifications Copyright 2023 AntGroups, Inc.

// Copyright (c) Tsinghua Statistical Artificial Intelligence & Learning Group.
// SPDX-License-Identifier: Apache-2.0

// Cuda kernels for quantization and mixed-precision packing.

#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)
#define fmax(a, b) ((a) > (b) ? (a): (b))
#define fmin(a, b) ((a) < (b) ? (a): (b))

using torch::IntArrayRef;
using torch::Tensor;

/**************************************************/
/***** Pack/Unpack Absmax Linear Quantization *****/
/**************************************************/
// Pack float16/32 data into int8 bit stream, for bits <= 8
template<typename scalar_t>
__global__ void pack_absmax_linear_8bit_kernel(
    int32_t bits,
    const scalar_t* __restrict__ data,
    const scalar_t* __restrict__ absmax,
    int8_t* __restrict__ packed,
    std::pair<uint64_t, uint64_t> seeds) {
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const float B = (1 << (bits - 1)) - 1;
  const int32_t mask = (1 << bits) - 1;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  for (int i = 0; i < workint_per_thread; i++) {
    uint8_t local_packed = 0;
    int64_t global_int_id = global_thread_id * workint_per_thread + i;
    for (int j = 0; j < work_per_int; j++) {
      const int64_t id = global_thread_id * work_per_thread \
        + i * work_per_int + j;
      const float noise = curand_uniform(&state);
      const int32_t val = __float2int_rn(fmax(fmin((data[id] \
        / absmax[group_id]) * B + noise - 0.5, B), -B));
      local_packed |= ((val & mask) << (j * bits));
    }

    packed[global_int_id] = local_packed;
  }
}

template<typename scalar_t>
__global__ void print_kernel(int32_t bits,
                             const scalar_t* __restrict__ data,
                             const scalar_t* __restrict__ absmax,
                             int8_t* __restrict__ packed,
                             std::pair<uint64_t, uint64_t> seeds) {
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const float B = (1 << (bits - 1)) - 1;
  const int32_t mask = (1 << bits) - 1;

  printf("group id: %d, thread id: %d, global id: %d\n", \
    group_id, d, global_thread_id);
  printf("data: %lf\n", data[global_thread_id * work_per_thread + 1]);
}

Tensor pack_absmax_linear_8bit_cuda(Tensor data,
                                    Tensor absmax,
                                    int bits,
                                    bool stochastic) {
  int64_t num_groups = data.size(0);
  int64_t group_size = data.size(1);

  // Compute total bits
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int * workint_per_thread;
  TORCH_CHECK(8 % bits == 0);

  int64_t total_bits = (int64_t)bits * (num_groups * group_size);
  auto options = torch::TensorOptions().dtype(
      torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8, }, options);

  // Random number generator
  int threads = group_size;
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(
      at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }
  TORCH_CHECK(stochastic);

  // Call pack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.scalar_type(), "pack_absmax_linear_8bit", ([&] {
  pack_absmax_linear_8bit_kernel<scalar_t> \
  <<<num_groups, group_size/work_per_thread>>>(
      bits,
      data.data_ptr<scalar_t>(),
      absmax.data_ptr<scalar_t>(),
      packed.data_ptr<int8_t>(),
      rng_engine_inputs);
  }));

  return packed;
}

// Pack float16/32 data into int8 bit stream, for 8 < bits <= 16
template<typename scalar_t>
__global__ void pack_absmax_linear_16bit_kernel(
    int32_t bits,
    const scalar_t* __restrict__ data,
    const scalar_t* __restrict__ absmax,
    int8_t* __restrict__ packed,
    std::pair<uint64_t, uint64_t> seeds,
    int64_t group_size) {
  const int64_t group_id = blockIdx.x;
  const int64_t d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int workbit_per_thread = 64;
  const int work_per_thread = workbit_per_thread / bits;
  const uint8_t packed8_mask = 0xff;
  const int B = (1 << (bits - 1)) - 1;
  const int64_t mask = (1 << bits) - 1;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  uint64_t local_packed = 0;
  for (int i = 0; i < work_per_thread; i++) {
    if (d * work_per_thread + i >= group_size)
      break;
    const int64_t data_id = group_id * group_size + d * work_per_thread + i;
    const float noise = curand_uniform(&state);
    const float x = data[data_id] / absmax[group_id];
    // ensure positivity of 'val': [0, 2B],
    // which was not introduced in 8-bit kernel
    const int64_t val = __float2int_rn(fmax(fmin(x \
        * B + noise - 0.5, static_cast<float>(B)), -static_cast<float>(B))) + B;
    local_packed |= ((val & mask) << (i * bits));
  }

  for (int i = 0; i < 8; i++) {
    const int64_t global_int_id = global_thread_id * 8 + i;
    uint8_t local_packed8 = (local_packed >> (i << 3)) & packed8_mask;
    packed[global_int_id] = local_packed8;
  }
}

Tensor pack_absmax_linear_16bit_cuda(Tensor data,
                                     Tensor absmax,
                                     int bits,
                                     bool stochastic) {
  int64_t num_groups = data.size(0);
  int64_t group_size = data.size(1);

  // Compute total bits
  const int workbit_per_thread = 64;
  const int work_per_thread = workbit_per_thread / bits;
  int64_t threads_num = (group_size + work_per_thread - 1) / work_per_thread;
  TORCH_CHECK(bits > 8);
  TORCH_CHECK(bits <= 16);

  int64_t total_bits = num_groups * threads_num * workbit_per_thread;
  auto options = torch::TensorOptions().dtype(
      torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits) / 8, }, options);

  // Random number generator
  int threads = group_size;
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(
      at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }
  TORCH_CHECK(stochastic);

  // Call pack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.scalar_type(), "pack_absmax_linear_16bit", ([&] {
  pack_absmax_linear_16bit_kernel<scalar_t><<<num_groups, threads_num>>>(
      bits,
      data.data_ptr<scalar_t>(),
      absmax.data_ptr<scalar_t>(),
      packed.data_ptr<int8_t>(),
      rng_engine_inputs,
      group_size);
  }));

  return packed;
}

Tensor pack_absmax_linear_cuda(Tensor data,
                               Tensor absmax,
                               int bits,
                               bool stochastic) {
  if (bits <= 8) {
    return pack_absmax_linear_8bit_cuda(data, absmax, bits, stochastic);
  } else {
    return pack_absmax_linear_16bit_cuda(data, absmax, bits, stochastic);
  }
}

// Unpack int8 bit stream to float16/32 data, for bits <= 8
template<typename scalar_t>
__global__ void unpack_absmax_linear_8bit_kernel(
    int32_t bits,
    const int8_t* __restrict__ data,
    const scalar_t* __restrict__ absmax,
    scalar_t* __restrict__ unpacked) {
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const scalar_t B = (1 << (bits - 1)) - 1;
  const int8_t mask = (1 << bits) - 1;

  for (int i = 0; i < workint_per_thread; i++) {
    int64_t global_int_id = global_thread_id * workint_per_thread + i;
    const int8_t local_packed = data[global_int_id];
    for (int j = 0; j < work_per_int; j++) {
      const int64_t id = global_thread_id \
        * work_per_thread + i * work_per_int + j;
      const int8_t unsigned_val = (local_packed >> (j * bits)) & mask;
      const int8_t val = ((unsigned_val > static_cast<int>(B)) ? \
        (unsigned_val | (~mask)) : unsigned_val);
      unpacked[id] = ((scalar_t)val) * (absmax[group_id] / B);
    }
  }
}

Tensor unpack_absmax_linear_8bit_cuda(Tensor data,
                                      int bits,
                                      Tensor absmax,
                                      int64_t num_groups,
                                      int64_t group_size) {
  auto options = torch::TensorOptions().dtype(
      absmax.dtype()).device(data.device());
  Tensor unpacked = torch::empty({num_groups, group_size}, options);

  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int * workint_per_thread;
  TORCH_CHECK(8 % bits == 0);

  // Call unpack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      absmax.scalar_type(), "unpack_absmax_linear_8bit", ([&] {
    unpack_absmax_linear_8bit_kernel<scalar_t> \
    <<<num_groups, group_size/work_per_thread>>>(
      bits,
      data.data_ptr<int8_t>(),
      absmax.data_ptr<scalar_t>(),
      unpacked.data_ptr<scalar_t>());
  }));

  return unpacked;
}

// Unpack int8 bit stream to float16/32 data, for 8 < bits <= 16
template<typename scalar_t>
__global__ void unpack_absmax_linear_16bit_kernel(
    int32_t bits,
    const int8_t* __restrict__ data,
    const scalar_t* __restrict__ absmax,
    scalar_t* __restrict__ unpacked,
    int64_t group_size) {
  const int64_t group_id = blockIdx.x;
  const int64_t d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int workbit_per_thread = 64;
  const int work_per_thread = workbit_per_thread / bits;
  const int B = (1 << (bits - 1)) - 1;
  const uint8_t packed8_mask = 0xff;
  const int64_t val_mask = (1 << bits) - 1;

  uint64_t local_packed = 0;
  for (int i = 0; i < 8; i++) {
    const int64_t global_int_id = global_thread_id * 8 + i;
    uint64_t local_packed8 = (uint64_t)(
        data[global_int_id] & packed8_mask) << (i << 3);
    local_packed |= local_packed8;
  }

  for (int i = 0; i < work_per_thread; i++) {
    if (d * work_per_thread + i >= group_size)
      break;
    const int64_t data_id = group_id * group_size + d * work_per_thread + i;
    const int64_t q_val_nonneg = (local_packed >> (i * bits)) & val_mask;
    unpacked[data_id] = (scalar_t)((q_val_nonneg - B)) * (absmax[group_id] / B);
  }
}

Tensor unpack_absmax_linear_16bit_cuda(Tensor data,
                                      int bits,
                                      Tensor absmax,
                                      int64_t num_groups,
                                      int64_t group_size) {
  auto options = torch::TensorOptions().dtype(
      absmax.dtype()).device(data.device());
  Tensor unpacked = torch::empty({num_groups, group_size}, options);

  const int workbit_per_thread = 64;
  const int work_per_thread = workbit_per_thread / bits;
  int64_t threads_num = (group_size + work_per_thread - 1) / work_per_thread;
  TORCH_CHECK(bits > 8);
  TORCH_CHECK(bits <= 16);

  // Call unpack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      absmax.scalar_type(), "unpack_absmax_linear_16bit", ([&] {
    unpack_absmax_linear_16bit_kernel<scalar_t><<<num_groups, threads_num>>>(
      bits,
      data.data_ptr<int8_t>(),
      absmax.data_ptr<scalar_t>(),
      unpacked.data_ptr<scalar_t>(),
      group_size);
  }));

  return unpacked;
}

Tensor unpack_absmax_linear_cuda(Tensor data,
                                 int bits,
                                 Tensor absmax,
                                 int64_t num_groups,
                                 int64_t group_size) {
  if (bits <= 8) {
    return unpack_absmax_linear_8bit_cuda(
        data, bits, absmax, num_groups, group_size);
  } else {
    return unpack_absmax_linear_16bit_cuda(
        data, bits, absmax, num_groups, group_size);
  }
}



/******************************************************/
/***** Pack/Unpack Absmax Non-Linear Quantization *****/
/******************************************************/
template<bool STOCHASTIC>
__device__ __forceinline__ int quantize_bsearch(const float* __restrict__ qmap,
                                                    int bits,
                                                    float x,
                                                    float noise) {
    int lo = 0;
    int hi = 1 << bits;

    if (x <= qmap[lo])
      return lo;
    if (qmap[hi - 1] <= x)
      return (hi - 1);

    while (lo < hi) {
      int mi = (lo + hi) >> 1;
      if (qmap[mi] <= x)
        lo = mi + 1;
      else
        hi = mi;
    }

    int rank = 0;
    if (STOCHASTIC) {
      float proba = (x - qmap[lo - 1]) / (qmap[lo] - qmap[lo - 1]);
      int flag = __float2int_rn(proba + noise - 0.5f);
      rank = (flag) ? lo : lo - 1;
    } else {
      float mid_val = (qmap[lo - 1] + qmap[lo]) * 0.5f;
      rank = (mid_val < x) ? lo : lo - 1;
    }
    return rank;
}

// Pack float16/32 data into int8 bit stream, for bits < 8 and 8 % bit == 0
template<typename scalar_t, bool STOCHASTIC>
__global__ void pack_nonlinear_4bit_kernel(int32_t bits,
                                          const scalar_t* __restrict__ data,
                                          const float* __restrict__ qmap,
                                          int8_t* __restrict__ packed,
                                          std::pair<uint64_t, uint64_t> seeds) {
  const int group_id = blockIdx.x;
  const int id_in_group = threadIdx.x;
  const int64_t global_id = group_id * blockDim.x + id_in_group;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const int8_t mask = (1 << bits) - 1;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_id, seeds.second, &state);

  for (int i = 0; i < workint_per_thread; i++) {
    uint8_t local_packed = 0;
    int64_t packed_id = global_id * workint_per_thread + i;
    for (int j = 0; j < work_per_int; j++) {
      const int64_t data_id = global_id \
        * work_per_thread + i * work_per_int + j;
      const float noise = curand_uniform(&state);
      const float x = data[data_id];
      const uint8_t qx = (uint8_t)quantize_bsearch<STOCHASTIC>(
          qmap, bits, x, noise);
      local_packed |= ((qx & mask) << (j * bits));
    }

    packed[packed_id] = local_packed;
  }
}

Tensor pack_nonlinear_4bit_cuda(Tensor data,
                                Tensor qmap,
                                int bits,
                                bool stochastic) {
  int64_t num_groups = data.size(0);
  int64_t group_size = data.size(1);

  // Compute total bits
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int * workint_per_thread;
  TORCH_CHECK(8 % bits == 0);
  TORCH_CHECK(group_size % work_per_thread == 0);

  int64_t total_bits = (int64_t)bits * (num_groups * group_size);
  auto options = torch::TensorOptions().dtype(
      torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8, }, options);

  // Random number generator
  int threads = group_size;
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(
      at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }

  // Call pack kernels
  if (stochastic) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.scalar_type(), "pack_nonlinear_4bit", ([&] {
    pack_nonlinear_4bit_kernel<scalar_t, true> \
    <<<num_groups, group_size/work_per_thread>>>(
        bits,
        data.data_ptr<scalar_t>(),
        qmap.data_ptr<float>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.scalar_type(), "pack_nonlinear_4bit", ([&] {
    pack_nonlinear_4bit_kernel<scalar_t, false> \
    <<<num_groups, group_size/work_per_thread>>>(
        bits,
        data.data_ptr<scalar_t>(),
        qmap.data_ptr<float>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs);
    }));
  }

  return packed;
}

// Pack float16/32 data into int8 bit stream, for bits in [5, 6, 7, (8)]
template<typename scalar_t, bool STOCHASTIC>
__global__ void pack_nonlinear_8bit_kernel(int32_t bits,
                                          const scalar_t* __restrict__ data,
                                          const float* __restrict__ qmap,
                                          int8_t* __restrict__ packed,
                                          std::pair<uint64_t, uint64_t> seeds) {
  const int group_id = blockIdx.x;
  const int id_in_group = threadIdx.x;
  const int64_t global_id = group_id * blockDim.x + id_in_group;
  const int work_per_thread = 4;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_id, seeds.second, &state);

  for (int i = 0; i < work_per_thread; i++) {
    const int64_t packed_id = global_id * work_per_thread + i;
    const float noise = curand_uniform(&state);
    const float x = data[packed_id];
    const uint8_t qx = (uint8_t)quantize_bsearch<STOCHASTIC>(
        qmap, bits, x, noise);
    packed[packed_id] = qx;
  }
}

Tensor pack_nonlinear_8bit_cuda(Tensor data,
                                Tensor qmap,
                                int bits,
                                bool stochastic) {
  int64_t num_groups = data.size(0);
  int64_t group_size = data.size(1);

  // Compute total bits
  const int storage_bits = 8;
  const int work_per_int = 1;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int * workint_per_thread;
  TORCH_CHECK(group_size % work_per_thread == 0);

  int64_t total_bits = (int64_t)storage_bits * (num_groups * group_size);
  auto options = torch::TensorOptions().dtype(
      torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8, }, options);

  // Random number generator
  int threads = group_size;
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(
      at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }

  // Call pack kernels
  if (stochastic) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    data.scalar_type(), "pack_nonlinear_8bit", ([&] {
    pack_nonlinear_8bit_kernel<scalar_t, true>\
    <<<num_groups, group_size/work_per_thread>>>(
        bits,
        data.data_ptr<scalar_t>(),
        qmap.data_ptr<float>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    data.scalar_type(), "pack_nonlinear_8bit", ([&] {
    pack_nonlinear_8bit_kernel<scalar_t, false> \
    <<<num_groups, group_size/work_per_thread>>>(
        bits,
        data.data_ptr<scalar_t>(),
        qmap.data_ptr<float>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs);
    }));
  }
  return packed;
}

Tensor pack_nonlinear_cuda(Tensor data,
                           Tensor qmap,
                           int bits,
                           bool stochastic) {
  if (8 % bits == 0 && bits < 8) {
    return pack_nonlinear_4bit_cuda(data, qmap, bits, stochastic);
  } else {  // bits <= 8
    return pack_nonlinear_8bit_cuda(data, qmap, bits, stochastic);
  }
}

// Unpack int8 bit stream to float16/32 data, for bits < 8 and 8 % bit == 0
template<typename scalar_t>
__global__ void unpack_nonlinear_4bit_kernel(int32_t bits,
                                            const int8_t* __restrict__ data,
                                            const float* __restrict__ qmap,
                                            scalar_t* __restrict__ unpacked) {
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const int8_t mask = (1 << bits) - 1;

  for (int i = 0; i < workint_per_thread; i++) {
    int64_t global_int_id = global_thread_id * workint_per_thread + i;
    const uint8_t local_packed = data[global_int_id];
    for (int j = 0; j < work_per_int; j++) {
      const int64_t id = global_thread_id \
        * work_per_thread + i * work_per_int + j;
      const uint8_t unsigned_val = (local_packed >> (j * bits)) & mask;
      unpacked[id] = (scalar_t)qmap[unsigned_val];
    }
  }
}

Tensor unpack_nonlinear_4bit_cuda(Tensor data,
                                  Tensor qmap,
                                  int bits,
                                  int64_t num_groups,
                                  int64_t group_size) {
  auto options = torch::TensorOptions().dtype(
    qmap.dtype()).device(data.device());
  Tensor unpacked = torch::empty({num_groups, group_size}, options);

  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int * workint_per_thread;
  TORCH_CHECK(8 % bits == 0);

  // Call unpack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      qmap.scalar_type(), "unpack_nonlinear_4bit", ([&] {
    unpack_nonlinear_4bit_kernel<scalar_t> \
    <<<num_groups, group_size/work_per_thread>>>(
      bits,
      data.data_ptr<int8_t>(),
      qmap.data_ptr<float>(),
      unpacked.data_ptr<scalar_t>());
  }));

  return unpacked;
}

// Unpack int8 bit stream to float16/32 data, for bits in [5, 6, 7, (8)]
template<typename scalar_t>
__global__ void unpack_nonlinear_8bit_kernel(int32_t bits,
                                            const int8_t* __restrict__ data,
                                            const float* __restrict__ qmap,
                                            scalar_t* __restrict__ unpacked) {
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = group_id * blockDim.x + d;
  const int work_per_thread = 4;

  for (int i = 0; i < work_per_thread; i++) {
    const int64_t global_int_id = global_thread_id * work_per_thread + i;
    const uint8_t local_packed = data[global_int_id];
    unpacked[global_int_id] = (scalar_t)qmap[local_packed];
  }
}

Tensor unpack_nonlinear_8bit_cuda(Tensor data,
                                  Tensor qmap,
                                  int bits,
                                  int64_t num_groups,
                                  int64_t group_size) {
  auto options = torch::TensorOptions().dtype(
    qmap.dtype()).device(data.device());
  Tensor unpacked = torch::empty({num_groups, group_size}, options);

  const int work_per_thread = 4;

  // Call unpack kernels
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      qmap.scalar_type(), "unpack_nonlinear_8bit", ([&] {
    unpack_nonlinear_8bit_kernel<scalar_t> \
    <<<num_groups, group_size/work_per_thread>>>(
        bits,
        data.data_ptr<int8_t>(),
        qmap.data_ptr<float>(),
        unpacked.data_ptr<scalar_t>());
  }));

  return unpacked;
}

Tensor unpack_nonlinear_cuda(Tensor data,
                             Tensor qmap,
                             int bits,
                             int64_t num_groups,
                             int64_t group_size) {
  if (8 % bits == 0 && bits < 8) {
    return unpack_nonlinear_4bit_cuda(data, qmap, bits, num_groups, group_size);
  } else {  // bits <= 8
    return unpack_nonlinear_8bit_cuda(data, qmap, bits, num_groups, group_size);
  }
}

