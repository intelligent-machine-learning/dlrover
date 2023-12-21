// Modifications Copyright 2023 AntGroups, Inc.

// Copyright (c) Tsinghua Statistical Artificial Intelligence & Learning Group.
// SPDX-License-Identifier: Apache-2.0

// Cuda operators for quantization and mixed-precision packing

#include <torch/extension.h>
#include <torch/torch.h>

#include "quantization_optimizer.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;

// Declarations for functions in quantization_kernel.cu
// Pack and unpack
Tensor pack_absmax_linear_cuda(
    Tensor data, Tensor absmax, int bits, bool stochastic);
Tensor unpack_absmax_linear_cuda(
    Tensor data, int bits, Tensor absmax, \
    int64_t num_groups, int64_t group_size);
Tensor pack_nonlinear_cuda(
    Tensor data, Tensor qmap, int bits, bool stochastic);
Tensor unpack_nonlinear_cuda(
    Tensor data, Tensor qmap, int bits, int64_t num_groups, int64_t group_size);

// Pack/Unpack with absmax linear quantization
Tensor pack_absmax_linear(Tensor data,
                          Tensor absmax,
                          int bits,
                          bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 2);
  CHECK_CUDA_TENSOR_DIM_FLOAT(absmax, 2);

  return pack_absmax_linear_cuda(data, absmax, bits, stochastic);
}

Tensor unpack_absmax_linear(Tensor data,
                            int bits,
                            Tensor absmax,
                            int64_t num_groups,
                            int64_t group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(absmax, 2);

  return unpack_absmax_linear_cuda(data, bits, absmax,
                                   num_groups, group_size);
}

// Pack/Unpack with nonlinear quantization
Tensor pack_nonlinear(Tensor data,
                      Tensor qmap,
                      int bits,
                      bool stochastic) {
  TORCH_CHECK(bits <= 8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 2);
  CHECK_CUDA_TENSOR_DIM_FLOAT(qmap, 1);

  return pack_nonlinear_cuda(data, qmap, bits, stochastic);
}

Tensor unpack_nonlinear(Tensor data,
                        Tensor qmap,
                        int bits,
                        int64_t num_groups,
                        int64_t group_size) {
  TORCH_CHECK(bits <= 8);
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(qmap, 1);

  return unpack_nonlinear_cuda(data, qmap, bits,
                               num_groups, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_absmax_linear", &pack_absmax_linear);
  m.def("unpack_absmax_linear", &unpack_absmax_linear);
  m.def("pack_nonlinear", &pack_nonlinear);
  m.def("unpack_nonlinear", &unpack_nonlinear);
}
