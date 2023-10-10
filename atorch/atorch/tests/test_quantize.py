# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest

import torch

from atorch.ops.accelerator import get_accelerator

# if not atorch.ops.__compatible_ops__[QuantizerBuilder.NAME]:
#     pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None


def run_quantize(activations, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        from atorch.ops.op_builder import QuantizerBuilder

        inference_module = QuantizerBuilder().load()

    return inference_module.quantize(
        activations,
        num_groups,
        q_bits,
        inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric,
    )


def run_dequantize(activations, params, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        from atorch.ops.op_builder import QuantizerBuilder

        inference_module = QuantizerBuilder().load()
    return inference_module.dequantize(
        activations,
        params,
        num_groups,
        q_bits,
        inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric,
    )


def get_q_props(q_bits):
    q_range = 2**q_bits
    q_min = -(2 ** (q_bits - 1))
    q_max = 2 ** (q_bits - 1) - 1

    q_min = torch.IntTensor([q_min]).to(device=get_accelerator().device_name())
    q_max = torch.IntTensor([q_max]).to(device=get_accelerator().device_name())
    return q_range, q_max, q_min


def get_scale_zero_point(q_bits, is_symmetric_quant, max, min, absmax, scales=None, zero_points=None):

    q_range, q_max, q_min = get_q_props(q_bits)

    if is_symmetric_quant:
        scale = torch.empty_like(absmax)
        for i, x in enumerate(absmax):
            scale[i] = torch.ones_like(x) if x == 0 else q_range / (2 * x)
        zero_point = torch.zeros(scale.shape, dtype=torch.float32, device=get_accelerator().device_name())
    else:
        scale = torch.empty_like(max)
        for i, x in enumerate(max):
            scale[i] = torch.ones_like(x) if max[i] == min[i] else q_range / (max[i] - min[i])
        zero_point = q_min - (min * scale)

    return scale, zero_point


def int4x2to2xint4(int4X2tensor):
    high = int4X2tensor >> 4
    low = (int4X2tensor << 4) >> 4
    return torch.stack((high, low), dim=-1).flatten()


def run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups):

    # Reference implementation
    # https://pytorch.org/docs/stable/quantization-support.html

    activations_ref = activations_ref.reshape(num_groups, -1).to(dtype=torch.float32)

    max_abs_activations_ref = torch.amax(torch.abs(activations_ref), dim=-1).view(num_groups, -1)
    max_activations_ref = torch.amax(activations_ref, dim=-1).view(num_groups, -1)
    min_activations_ref = torch.amin(activations_ref, dim=-1).view(num_groups, -1)

    _, q_max, q_min = get_q_props(q_bits)

    scale, zero_point = get_scale_zero_point(
        q_bits, is_symmetric_quant, max_activations_ref, min_activations_ref, max_abs_activations_ref
    )

    data_f = activations_ref * scale

    if not is_symmetric_quant:
        data_f = data_f + zero_point

    data_i32 = torch.round(data_f).to(dtype=torch.int32)

    data_i32 = torch.minimum(torch.maximum(data_i32, q_min.expand_as(data_i32)), q_max.expand_as(data_i32))
    data_i8 = data_i32.to(dtype=torch.int8)

    scales = (1.0 / scale).reshape(-1, 1)
    offsets = zero_point.reshape(-1, 1)
    params = torch.cat((scales, offsets), dim=-1)

    return data_i8, params


def run_float_dequantize(q_bits, is_symmetric_quant, data_i8, params, num_groups):
    data_f = data_i8.reshape(num_groups, -1).to(dtype=torch.float32)

    scales = params[:, 0].reshape(-1, 1)
    offsets = params[:, 1].reshape(-1, 1)

    if not is_symmetric_quant:
        data_f = data_f - offsets
    else:
        assert offsets.allclose(torch.zeros_like(offsets))

    data_f = data_f * scales

    return data_f


class QuantizeTest(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_quantize(self):
        def callback(num_elems, num_groups, is_symmetric_quant, q_bits, directed_case):
            # fix seed
            torch.manual_seed(num_elems)

            if directed_case == "all_zeros":
                activations = torch.zeros(
                    (num_groups, num_elems), dtype=torch.float16, device=get_accelerator().device_name()
                )
            else:
                activations = torch.randn(
                    (num_groups, num_elems), dtype=torch.float16, device=get_accelerator().device_name()
                )
            activations_ref = activations.clone().detach()

            ref_out_tensor, ref_params = run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups)
            ref_dequantized_tensor = run_float_dequantize(
                q_bits, is_symmetric_quant, ref_out_tensor, ref_params, num_groups
            )
            # we need to convert the tensor to float64 to avoid overflow
            ref_quantization_error = torch.sum(torch.abs((activations_ref - ref_dequantized_tensor).to(torch.float64)))

            out_tensor, out_params = run_quantize(activations, num_groups, q_bits, is_symmetric_quant)
            dequantized_tensor = run_dequantize(out_tensor, out_params, num_groups, q_bits, is_symmetric_quant)
            assert torch.all(torch.isfinite(dequantized_tensor))

            quantization_error = torch.sum(torch.abs((activations - dequantized_tensor).to(torch.float64)))

            assert quantization_error <= ref_quantization_error * 1.05

        for _num_groups in [1, 13, 512]:
            for _num_elems in [8, 16, 32, 64, 128, 256, 4096, 8192, 12288, 16384]:
                for _is_symmetric_quant in True, False:
                    for _q_bits in [4, 8]:
                        for _directed_case in ["all_zeros", None]:
                            callback(_num_elems, _num_groups, _is_symmetric_quant, _q_bits, _directed_case)


class CUDAQuantizeTest(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_cuda_quantizer(self):
        num_elems = 4096
        num_groups = 512
        from atorch.ops.quantizer import CUDAQuantizer

        cuda_quantizer = CUDAQuantizer()

        torch.manual_seed(num_elems)

        activations = torch.randn((num_groups, num_elems), dtype=torch.float16, device=get_accelerator().device_name())

        activations_ref = activations.clone().detach()

        ref_out_tensor_1, ref_params_1 = cuda_quantizer.quantize(activations)
        ref_dequantized_tensor_1 = cuda_quantizer.dequantize(ref_out_tensor_1, ref_params_1)

        # we need to convert the tensor to float64 to avoid overflow
        ref_quantization_error_1 = torch.sum(torch.abs((activations_ref - ref_dequantized_tensor_1).to(torch.float64)))

        ref_out_tensor_2, ref_params_2 = run_float_quantize(8, True, activations_ref, num_groups)
        ref_dequantized_tensor_2 = run_float_dequantize(8, True, ref_out_tensor_2, ref_params_2, num_groups)
        ref_quantization_error_2 = torch.sum(torch.abs((activations_ref - ref_dequantized_tensor_2).to(torch.float64)))

        assert ref_quantization_error_1 <= ref_quantization_error_2 * 1.05


if __name__ == "__main__":
    unittest.main()
