# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class QuantizationOptimizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "ATorch_BUILD_QUANTIZATION_OPTIMIZER"
    NAME = "quantization_optimizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"atorch.ops.quantization_optimizer.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/quantization/quantization_optimizer.cc",
            "csrc/quantization/quantization_optimizer.cu",
        ]

    def include_paths(self):
        return ["csrc/includes"]

    def extra_ldflags(self):
        return ["-lcurand"]
