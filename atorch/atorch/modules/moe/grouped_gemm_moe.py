try:
    import grouped_gemm as gg
except ImportError:
    gg = None

try:
    import megablocks.ops as ops
except (ImportError, RuntimeError):
    ops = None

import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp.autocast_mode import _cast, autocast

from atorch.modules.moe.ops import bias_gather_add


# patch fn to handle autocast
def _cast_fn(fn):
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        if torch.is_autocast_enabled():
            cur_dtype = torch.get_autocast_gpu_dtype()
            with autocast(enabled=False):
                return fn(*_cast(args, cur_dtype), **_cast(kwargs, cur_dtype))
        else:
            return fn(*args, **kwargs)

    return new_fn


if gg is not None:
    gg.ops.gmm = _cast_fn(gg.ops.gmm)


class SwiGLUActivatition(nn.Module):
    def forward(self, input):
        input = torch.chunk(input, 2, dim=-1)
        return F.silu(input[0]) * input[1]


class Grouped_GEMM_MoE(torch.nn.Module):
    """
    A Mixture of Experts (MoE) with grouped GEMM
    """

    def __init__(
        self,
        hidden_size,
        expert_intermediate_size,
        output_dropout_prob,
        num_experts,
        topk,
        use_swiglu=False,
        use_bias=True,
        initializer_range=0.02,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.output_dropout_prob = output_dropout_prob
        self.num_experts = num_experts
        self.topk = topk
        self.use_swiglu = use_swiglu
        self.use_bias = use_bias

        # rm `fine_grained_factor`, make sure num_experts, top_k have been `update_moe_config`
        # and pass the updated `expert_intermediate_size`

        self.activation = SwiGLUActivatition() if self.use_swiglu else torch.nn.functional.gelu

        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden_size, self.expert_intermediate_size * (2 if self.use_swiglu else 1)
            )
        )
        self.w2 = torch.nn.Parameter(torch.empty(self.num_experts, self.expert_intermediate_size, self.hidden_size))
        if self.use_bias:
            self.b1 = torch.nn.Parameter(
                torch.empty(self.num_experts, self.expert_intermediate_size * (2 if self.use_swiglu else 1))
            )
            self.b2 = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

        self.dropout = torch.nn.Dropout(self.output_dropout_prob)

        # reset gg weight bias
        self.w1.data.normal_(mean=0.0, std=initializer_range)
        self.w2.data.normal_(mean=0.0, std=initializer_range)
        if self.use_bias:
            self.b1.data.zero_()
            self.b2.data.zero_()

        # megablocks gather scatter
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

    def permutation_hidden_states(self, hidden_states, top_experts):
        """refer to megablocks"""
        top_expert = top_experts.flatten().int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins

        permuted_hidden_states = ops.gather(hidden_states, indices, bin_ids, bins, self.topk)
        return permuted_hidden_states, indices, bin_ids, bins, tokens_per_expert

    def forward(self, hidden_states, expert_weights, top_experts):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        expert_weights = expert_weights.flatten()

        # permute hidden states
        permuted_hidden_states, indices, bin_ids, bins, tokens_per_expert = self.permutation_hidden_states(
            hidden_states, top_experts
        )

        # compute moe
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
        fc1_output = gg.ops.gmm(permuted_hidden_states, self.w1, tokens_per_expert, trans_b=False)
        if self.use_bias:
            fc1_output = bias_gather_add(fc1_output, self.b1, bin_ids)
        intermediate_parallel = self.activation(fc1_output)
        fc2_output = gg.ops.gmm(intermediate_parallel, self.w2, tokens_per_expert, trans_b=False)
        if self.use_bias:
            fc2_output = bias_gather_add(fc2_output, self.b2, bin_ids)
        expert_output = self.dropout(fc2_output)

        # unpermute expert outpute
        output = ops.scatter(expert_output, indices, bin_ids, expert_weights, bins, self.topk).view(self.hidden_shape)

        return output
