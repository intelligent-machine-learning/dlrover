# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Re-implementation of a LoRA adapter.
References: 1) LoRA: Low-Rank Adaptation of Large Language Models -
https://arxiv.org/abs/2106.09685 2) minLoRA - https://github.com/cccntu/minLoRA
Leverages the parametrizations feature from pytorch. This allows us to add the
LoRA matrices to the weights during the forward pass rather than computing the
modified forward pass explicitly, i.e., we compute (W + BA)x rather than Wx +
BAx. 3) LoRA - https://github.com/microsoft/LoRA This repo contains the source
code of the Python package loralib, which implements the LoRA algorithm.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoraLinear(nn.Module):
    def __init__(self, fan_in, fan_out, rank=4, dropout_p=0.0, alpha=1.0):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.rank = rank
        self.dropout_p = dropout_p

        self.lora_a = nn.Parameter(torch.zeros(rank, fan_in))
        self.lora_b = nn.Parameter(torch.zeros(fan_out, rank))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, weight):
        return (
            weight
            + torch.matmul(self.lora_b, self.dropout(self.lora_a))
            * self.scaling
        )


def apply_lora(
    model: nn.Module,
    targets=["wq", "wk", "wo", "wv"],
    rank=8,
    dropout=0.0,
    alpha=1.0,
    verbose=False,
):

    for name, module in model.named_modules():
        if any(name.endswith(target) for target in targets) and hasattr(
            module, "weight"
        ):
            fan_out, fan_in = module.weight.shape
            parametrize.register_parametrization(
                module,
                "weight",
                LoraLinear(
                    fan_in,
                    fan_out,
                ),
            )
            if verbose:
                print(f"add lora to {name}")


def merge_lora(model):
    def _merge_lora(module):
        if type(module) in (nn.Linear, nn.Embedding) and hasattr(
            module, "parametrizations"
        ):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )

    model.apply(_merge_lora)


def tie_lora_weights(src, trg):
    """Tie the LoRA weights between two modules. Can be useful for tying
    embeddings to the final classifier."""
    if hasattr(src, "parametrizations") and hasattr(trg, "parametrizations"):
        trg.parametrizations.weight[0].lora_a = src.parametrizations.weight[
            0
        ].lora_a
        trg.parametrizations.weight[0].lora_b = src.parametrizations.weight[
            0
        ].lora_b
