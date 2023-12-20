// Modifications Copyright 2023 AntGroups, Inc.

// Copyright (c) Tsinghua Statistical Artificial Intelligence & Learning Group.
// SPDX-License-Identifier: Apache-2.0

#ifndef ATORCH_OPS_CSRC_INCLUDES_QUANTIZATION_OPTIMIZER_H_
#define ATORCH_OPS_CSRC_INCLUDES_QUANTIZATION_OPTIMIZER_H_

// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_TYPE(name, n_dim, type) \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!"); \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!"); \
  TORCH_CHECK(name.dim() == n_dim, \
    "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!"); \

// Helper for type check
#define CHECK_CUDA_TENSOR_TYPE(name, type) \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!"); \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!"); \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!"); \

// Helper for type check
#define CHECK_CUDA_TENSOR_FLOAT(name) \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!"); \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!"); \
  TORCH_CHECK(name.dtype() == torch::kFloat32 \
    || name.dtype() == torch::kFloat16, \
              "The type of " #name " is not kFloat32 or kFloat16!"); \

// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_FLOAT(name, n_dim) \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!"); \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!"); \
  TORCH_CHECK(name.dim() == n_dim, \
    "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == torch::kFloat32 \
    || name.dtype() == torch::kFloat16, \
              "The type of " #name " is not kFloat32 or kFloat16!"); \

#endif  // ATORCH_OPS_CSRC_INCLUDES_QUANTIZATION_OPTIMIZER_H_
