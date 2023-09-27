// Modifications Copyright 2023 AntGroups, Inc.
// ATorch Team

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cassert>
#include <vector>

#include "quantization.h"

std::vector<at::Tensor> quantize_kernel(at::Tensor& input_vals, int groups,
                                        int numBits, quantize::Type quantType) {
  auto dtype = at::kFloat;
  auto params_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);
  const int param_elems = (quantize::requires_offset(quantType)) ? 2 : 1;
  auto params = torch::empty({groups, param_elems}, params_options);

  auto output_options = at::TensorOptions()
                            .dtype(at::kChar)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);

  auto output_sizes = input_vals.sizes().vec();
  output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
  auto output = torch::empty(output_sizes, output_options);

  const int elems_per_group = at::numel(input_vals) / groups;

  launch_quant(reinterpret_cast<int8_t*>(output.data_ptr()),
               reinterpret_cast<float*>(params.data_ptr()),
               reinterpret_cast<__half*>(input_vals.data_ptr()), groups,
               elems_per_group, numBits, quantType,
               at::cuda::getCurrentCUDAStream());

  return {output, params};
}

template <typename T>
at::Tensor dequantize(at::Tensor& quantized_data, at::Tensor& params,
                      int groups, int num_bits, quantize::Type quant_type) {
  auto dtype =
      (std::is_same<T, float>::value) ? torch::kFloat32 : torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);

  auto output_sizes = quantized_data.sizes().vec();
  output_sizes[output_sizes.size() - 1] *= num_bits == 8 ? 1 : 2;
  auto output = torch::empty(output_sizes, output_options);

  const int total_elems = at::numel(output);
  const int elems_per_group = total_elems / groups;

  launch_dequantize_kernel(
      reinterpret_cast<T*>(output.data_ptr()),
      reinterpret_cast<const int8_t*>(quantized_data.data_ptr()),
      reinterpret_cast<const float*>(params.data_ptr()), quant_type, num_bits,
      elems_per_group, total_elems, at::cuda::getCurrentCUDAStream());

  return output;
}

std::vector<at::Tensor> swizzle_quant(at::Tensor& input_vals, int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size, int nodes,
                                         int devices_per_node) {
  auto scales_options = at::TensorOptions()
                            .dtype(at::kFloat)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);
  const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
  auto scales = torch::empty({groups, scales_elems}, scales_options);

  auto output_options = at::TensorOptions()
                            .dtype(at::kChar)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);

  const int quantization_scalar = 8 / num_bits;
  const int compressed_vals = at::numel(input_vals) / quantization_scalar;

  auto output = torch::empty({compressed_vals}, output_options);
  const int elems_per_group = at::numel(input_vals) / groups;

  launch_swizzled_quant(reinterpret_cast<int8_t*>(output.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<__half*>(input_vals.data_ptr()),
                        num_bits, quant_type, groups, elems_per_group,
                        pipeline_size, nodes, devices_per_node,
                        at::cuda::getCurrentCUDAStream());

  return {output, scales};
}

std::vector<at::Tensor> quantized_reduction(at::Tensor& input_vals,
                                            at::Tensor& input_scales,
                                            int in_groups, int out_groups,
                                            int num_bits,
                                            quantize::Type quant_type,
                                            int devices_per_node) {
  auto scales_options = at::TensorOptions()
                            .dtype(at::kFloat)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);
  const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
  auto scales = torch::empty({out_groups, scales_elems}, scales_options);

  auto output_options = at::TensorOptions()
                            .dtype(at::kChar)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);

  std::vector<long int> sz(input_vals.sizes().begin(),
                           input_vals.sizes().end());
  sz[sz.size() - 1] = sz.back() / devices_per_node;  // num of GPU per nodes
  const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
  auto output = torch::empty(sz, output_options);

  const int elems_per_in_group =
      elems_per_in_tensor / (in_groups / devices_per_node);
  const int elems_per_out_group = elems_per_in_tensor / out_groups;

  launch_dequant_reduce(reinterpret_cast<int8_t*>(output.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<const int8_t*>(input_vals.data_ptr()),
                        reinterpret_cast<const float*>(input_scales.data_ptr()),
                        devices_per_node,
                        num_bits, quant_type, out_groups, elems_per_out_group,
                        elems_per_in_tensor, in_groups / devices_per_node,
                        elems_per_in_group, at::cuda::getCurrentCUDAStream());
  return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::enum_<quantize::Type>(m, "QuantizationType")
      .value("Symmetric", quantize::Type::Symmetric)
      .value("Asymmetric", quantize::Type::Asymmetric)
      .export_values();
  m.def("quantize", &quantize_kernel, py::arg("input_vals").noconvert(),
        py::arg("groups").noconvert(), py::arg("numBits").noconvert(),
        py::arg("quantType").noconvert());
  m.def("dequantize", &dequantize<__half>,
        py::arg("quantized_data").noconvert(), py::arg("params").noconvert(),
        py::arg("groups").noconvert(), py::arg("num_bits").noconvert(),
        py::arg("quant_type").noconvert());
  m.def("dequantize_fp32", &dequantize<float>,
        py::arg("quantized_data").noconvert(), py::arg("params").noconvert(),
        py::arg("groups").noconvert(), py::arg("num_bits").noconvert(),
        py::arg("quant_type").noconvert());
  m.def("swizzle_quant", &swizzle_quant, py::arg("input_vals").noconvert(),
        py::arg("groups").noconvert(), py::arg("num_bits").noconvert(),
        py::arg("quant_type").noconvert(),
        py::arg("pipeline_sizes").noconvert(), py::arg("nodes").noconvert(),
        py::arg("devices_per_node").noconvert());
  m.def("quantized_reduction", &quantized_reduction,
        py::arg("input_vals").noconvert(), py::arg("input_scales").noconvert(),
        py::arg("in_groups").noconvert(), py::arg("out_groups").noconvert(),
        py::arg("num_bits").noconvert(), py::arg("quant_type").noconvert(),
        py::arg("devices_per_node").noconvert());
}
