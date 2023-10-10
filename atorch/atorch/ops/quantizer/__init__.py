# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

from atorch.ops.accelerator import get_accelerator
from atorch.ops.op_builder import QuantizerBuilder  # type: ignore


class CUDAQuantizer(object):
    async_flag = True
    target_group_size = 8000  # the optimal size is 4k, so we set the target to be below 8k
    group_size_cache = dict()  # type: ignore
    quantizer_cuda_module = None

    def __init__(self):
        CUDAQuantizer.quantizer_cuda_module = QuantizerBuilder().load()

    def quantize(self, param, groups=None):
        # calculate the group size from the parameter numel
        if groups is None:
            try:
                groups = self.group_size_cache[param.numel()]
            except KeyError:
                groups = math.ceil(param.numel() / self.target_group_size)
                while groups < param.numel():
                    if param.numel() % (8 * groups) == 0:
                        break
                    groups += 1
                while True:
                    if (
                        param.numel() % (8 * groups * 2) == 0 and param.numel() / groups > self.target_group_size
                    ):  # hard limit of 16k group_size
                        groups *= 2
                    else:
                        break
                assert (
                    param.numel() % (8 * groups) == 0
                ), f"Qantized weight requires the number of weights be a multiple of 8. Yet {param.numel()} "
                "cannot be divided by 8*{groups}"
                assert param.numel() / groups < 16000, f"{param.numel()} / {groups} is larger than 16k"
                assert (
                    param.numel() > groups
                ), f"Adaptive grouping algorithm cannot find a group size for input tensor of size {param.numel()}"
                self.group_size_cache[param.numel()] = groups
        return CUDAQuantizer.quantizer_cuda_module.quantize(
            param.to(get_accelerator().device_name()), groups, 8, CUDAQuantizer.quantizer_cuda_module.Symmetric
        )

    def dequantize(self, quantized_param, scale):
        return CUDAQuantizer.quantizer_cuda_module.dequantize(
            quantized_param, scale, scale.numel(), 8, CUDAQuantizer.quantizer_cuda_module.Symmetric
        )
