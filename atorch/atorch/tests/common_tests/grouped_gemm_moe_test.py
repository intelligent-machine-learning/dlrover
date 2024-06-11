import copy
import os
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import Linear

from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE, SwiGLUActivatition


class MLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
    """

    def __init__(self, hidden_size, output_dropout_prob, intermediate_size=None, use_swiglu=False, use_bias=True):
        super(MLP, self).__init__()
        # Default project to 4h.
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        intermediate_size = int(intermediate_size)

        if use_swiglu:
            # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
            self.dense_h_to_4h = Linear(hidden_size, 2 * intermediate_size, bias=use_bias)
        else:
            self.dense_h_to_4h = Linear(hidden_size, intermediate_size, bias=use_bias)

        # Project back to h.
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, bias=use_bias)

        # self.activation_func = swiglu if use_swiglu else gelu
        self.activation_func = SwiGLUActivatition() if use_swiglu else torch.nn.functional.gelu
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class RefMoE(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, expert_intermediate_size, use_swiglu, use_bias):
        super().__init__()
        self.experts = nn.ModuleDict()
        for idx in range(num_experts):
            self.experts[f"expert_{idx}"] = MLP(
                hidden_size,
                0.0,
                intermediate_size=expert_intermediate_size,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
            )

    def forward(self, hidden_states, topk_weight, topk_idx):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        top_k = topk_weight.shape[-1]

        hidden_states = hidden_states.repeat_interleave(top_k, dim=0)
        y = torch.empty_like(hidden_states).to(
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else hidden_states.dtype
        )
        for i, expert in enumerate(self.experts.values()):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=len(topk_weight.shape) - 1)
        y = y.to(hidden_states.dtype)
        y = y.view(*orig_shape)
        return y


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
class TestGroupedGemmMoE(unittest.TestCase):

    seed = 1234

    def setUp(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    def test_grouped_gemm_moe(self):
        # for num_experts, hidden_size, expert_intermediate_size, topk in [[8, 3072, 12288, 2], [30, 3072, 3072, 6]]:
        for num_experts, hidden_size, expert_intermediate_size, topk in [[8, 256, 1024, 2], [30, 512, 512, 6]]:
            for use_swiglu in [False, True]:
                for use_bias in [False, True]:
                    self._test_grouped_gemm_moe(
                        num_experts, hidden_size, expert_intermediate_size, topk, use_swiglu, use_bias
                    )

    def sd_trans(self, ref_sd, num_experts, use_bias):
        new_sd = dict()
        w1 = torch.cat(
            [
                ref_sd[f"experts.expert_{eid}.dense_h_to_4h.weight"].transpose(0, 1).unsqueeze(0)
                for eid in range(num_experts)
            ]
        )
        w2 = torch.cat(
            [
                ref_sd[f"experts.expert_{eid}.dense_4h_to_h.weight"].transpose(0, 1).unsqueeze(0)
                for eid in range(num_experts)
            ]
        )
        if use_bias:
            b1 = torch.cat(
                [ref_sd[f"experts.expert_{eid}.dense_h_to_4h.bias"].unsqueeze(0) for eid in range(num_experts)]
            )
            b2 = torch.cat(
                [ref_sd[f"experts.expert_{eid}.dense_4h_to_h.bias"].unsqueeze(0) for eid in range(num_experts)]
            )
        new_sd["w1"] = w1
        new_sd["w2"] = w2
        if use_bias:
            new_sd["b1"] = b1
            new_sd["b2"] = b2
        return new_sd

    def assert_precision(self, fp32_ref, ref, GG, msg=None):
        ref_max = (fp32_ref - ref).abs().max()
        ref_mean = (fp32_ref - ref).abs().mean()
        GG_max = (fp32_ref - GG).abs().max()
        GG_mean = (fp32_ref - GG).abs().mean()
        # sometimes max larger than two times
        assert GG_max <= ref_max * 3, f"{msg} ref_max: {ref_max}, GG_max: {GG_max}"
        assert GG_mean <= ref_mean * 2, f"{msg} ref_mean: {ref_mean}, GG_mean: {GG_mean}"

    def _test_grouped_gemm_moe(self, num_experts, hidden_size, expert_intermediate_size, topk, use_swiglu, use_bias):
        print(
            f"num_experts {num_experts}, hidden_size {hidden_size}, expert_intermediate_size "
            f"{expert_intermediate_size}, topk {topk}, use_swiglu {use_swiglu}, use_bias {use_bias}"
        )
        GG_MoE = Grouped_GEMM_MoE(hidden_size, expert_intermediate_size, 0.0, num_experts, topk, use_swiglu, use_bias)
        ref_MoE = RefMoE(num_experts, hidden_size, expert_intermediate_size, use_swiglu, use_bias)
        fp32_ref_MoE = copy.deepcopy(ref_MoE)

        # align weight
        GG_MoE.load_state_dict(self.sd_trans(ref_MoE.state_dict(), num_experts, use_bias), strict=True)

        # to cuda
        fp32_ref_MoE.to(torch.cuda.current_device())
        ref_MoE.to(torch.cuda.current_device())
        GG_MoE.to(torch.cuda.current_device())

        # inputs, grad
        bs = 4
        seq_len = 4096
        inp = torch.randn(
            bs,
            seq_len,
            hidden_size,
            device=torch.cuda.current_device(),
            requires_grad=True,
        )
        ref_inp = inp.detach().clone()
        fp32_ref_inp = inp.detach().clone()
        expert_weights = torch.randn(
            bs,
            seq_len,
            topk,
            device=torch.cuda.current_device(),
        ).softmax(-1)
        top_experts = torch.randint(
            0,
            num_experts,
            (bs, seq_len, topk),
            device=torch.cuda.current_device(),
            dtype=torch.long,
        )
        rand_out_grad = torch.randn_like(inp)

        # fw bw
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ref_out = ref_MoE(ref_inp, expert_weights, top_experts)
            GG_out = GG_MoE(inp, expert_weights, top_experts)
        fp32_ref_out = fp32_ref_MoE(fp32_ref_inp, expert_weights, top_experts)
        ref_out.backward(rand_out_grad)
        GG_out.backward(rand_out_grad)
        fp32_ref_out.backward(rand_out_grad)

        # check precision
        self.assert_precision(fp32_ref_out, ref_out, GG_out, "output")
        ref_g = self.sd_trans({n: p.grad for n, p in ref_MoE.named_parameters()}, num_experts, use_bias)
        fp32_ref_g = self.sd_trans({n: p.grad for n, p in fp32_ref_MoE.named_parameters()}, num_experts, use_bias)
        for n, p in GG_MoE.named_parameters():
            self.assert_precision(fp32_ref_g[n], ref_g[n], p.grad, f"{n} grad")

        # time compare
        if os.environ.get("TIME_COMPARE") == "1":

            def fwbw(MoE):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = MoE(inp, expert_weights, top_experts)
                out.backward(rand_out_grad)

            def benchmark(MoE, msg):
                # warmup
                for _ in range(10):
                    fwbw(MoE)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(10):
                    fwbw(MoE)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                print(f"{msg} fwbw 10 times elapsed time: {elapsed_time_ms:.2f} ms")

            benchmark(ref_MoE, "Ref MoE")
            benchmark(GG_MoE, "GG MoE")
