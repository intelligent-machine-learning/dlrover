import copy
import os
import random
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import Linear

import atorch
import atorch.distributed
from atorch import npu  # noqa
from atorch.common.util_func import divide, find_free_port
from atorch.distributed.distributed import create_parallel_group
from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE, SwiGLUActivatition
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.moe_util import transpose_moe
from atorch.utils.version import torch_version

if torch.cuda.is_available():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def sd_trans(ref_sd, num_experts, use_bias):
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
        b1 = torch.cat([ref_sd[f"experts.expert_{eid}.dense_h_to_4h.bias"].unsqueeze(0) for eid in range(num_experts)])
        b2 = torch.cat([ref_sd[f"experts.expert_{eid}.dense_4h_to_h.bias"].unsqueeze(0) for eid in range(num_experts)])
    new_sd["w1"] = w1
    new_sd["w2"] = w2
    if use_bias:
        new_sd["b1"] = b1
        new_sd["b2"] = b2
    return new_sd


def assert_precision(fp32_ref, ref, GG, msg=None):
    ref_max = (fp32_ref - ref).abs().max()
    ref_mean = (fp32_ref - ref).abs().mean()
    GG_max = (fp32_ref - GG).abs().max()
    GG_mean = (fp32_ref - GG).abs().mean()
    # sometimes max larger than two times
    assert GG_max <= ref_max * 3, f"{msg} ref_max: {ref_max}, GG_max: {GG_max}"
    assert GG_mean <= ref_mean * 2, f"{msg} ref_mean: {ref_mean}, GG_mean: {GG_mean}"


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
    def __init__(self, num_experts, hidden_size, expert_intermediate_size, use_swiglu, use_bias, num_shared_experts=0):
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
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            shared_intermediate_size = expert_intermediate_size * num_shared_experts
            self.shared_experts = MLP(
                hidden_size,
                0.0,
                intermediate_size=shared_intermediate_size,
                use_swiglu=use_swiglu,
                use_bias=use_bias,
            )

    def forward(self, hidden_states, topk_weight, topk_idx):
        identity = hidden_states
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
        if self.num_shared_experts > 0:
            y = y + self.shared_experts(identity)
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

    @unittest.skipIf(is_torch_npu_available(), "MegaBlocks implementation_type not supported yet for npu")
    def test_grouped_gemm_moe_megablocks(self):
        # for num_experts, hidden_size, expert_intermediate_size, topk in [[8, 3072, 12288, 2], [30, 3072, 3072, 6]]:
        for num_experts, hidden_size, expert_intermediate_size, topk in [[8, 256, 1024, 2], [30, 512, 512, 6]]:
            for use_swiglu in [False, True]:
                for use_bias in [False, True]:
                    self._test_grouped_gemm_moe(
                        num_experts, hidden_size, expert_intermediate_size, topk, use_swiglu, use_bias
                    )

    def test_grouped_gemm_moe_merge_w1_v1(self):
        num_experts = 4
        hidden_size = 256
        expert_intermediate_size = 256
        bs = 2
        seq_len = 256
        device = torch.cuda.current_device()

        for num_experts in [1, 4]:
            torch.manual_seed(42 + num_experts)
            topk = 2 if num_experts > 2 else 1
            moe_module_merge = Grouped_GEMM_MoE(
                hidden_size,
                expert_intermediate_size,
                0.0,
                num_experts,
                topk,
                use_swiglu=True,
                use_bias=False,
                use_expert_parallelism=False,
                merge_w1_v1=True,
                transpose_w1=False,
                implementation_type="Megatron",
                token_dispatcher_type="MindSpeedAllGather",
            ).to(device)
            moe_module_not_merge = Grouped_GEMM_MoE(
                hidden_size,
                expert_intermediate_size,
                0.0,
                num_experts,
                topk,
                use_swiglu=True,
                use_bias=False,
                use_expert_parallelism=False,
                merge_w1_v1=False,
                transpose_w1=False,
                implementation_type="Megatron",
                token_dispatcher_type="MindSpeedAllGather",
            ).to(device)
            # sync parameters w1 -> (w1, v1), w2 ->w2
            with torch.no_grad():
                moe_module_not_merge.w2.copy_(moe_module_merge.w2)
                w1, v1 = torch.chunk(moe_module_merge.w1, 2, dim=-1)
                moe_module_not_merge.w1.copy_(w1)
                moe_module_not_merge.v1.copy_(v1)
            # create inputs
            inp = torch.randn(
                bs,
                seq_len,
                hidden_size,
                device=device,
                requires_grad=True,
            )
            expert_weights = torch.randn(
                bs,
                seq_len,
                topk,
                device=device,
            ).softmax(-1)

            top_experts = torch.randint(
                0,
                num_experts,
                (bs, seq_len, topk),
                device=device,
                dtype=torch.long,
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                merge_out = moe_module_merge(inp, expert_weights, top_experts)
                not_merge_out = moe_module_not_merge(inp, expert_weights, top_experts)
                merge_sum = torch.sum(merge_out)
                not_merge_sum = torch.sum(not_merge_out)
            merge_sum.backward()
            not_merge_sum.backward()
            self.assertTrue(torch.allclose(merge_out, not_merge_out, atol=1e-3))
            self.assertTrue(torch.allclose(moe_module_merge.w2.grad, moe_module_not_merge.w2.grad, atol=1e-3))
            with torch.no_grad():
                gw1, gv1 = torch.chunk(moe_module_merge.w1.grad, 2, dim=-1)
                self.assertTrue(torch.allclose(moe_module_not_merge.w1.grad, gw1, rtol=1e-2, atol=1e-2))
                self.assertTrue(torch.allclose(moe_module_not_merge.v1.grad, gv1, rtol=1e-2, atol=1e-2))

            # test no token
            inp = torch.empty([0, hidden_size], dtype=torch.float, device=device)
            tokens_per_expert = torch.zeros([num_experts], dtype=torch.int, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                res = moe_module_not_merge.compute_expert(inp, tokens_per_expert)
            self.assertEqual(res.shape[0], 0)

    def test_grouped_gemm_moe_transpose_w1(self):
        num_experts = 4
        hidden_size = 256
        expert_intermediate_size = 256
        bs = 2
        seq_len = 256
        device = torch.cuda.current_device()

        for num_experts in [1, 4]:
            torch.manual_seed(42 + num_experts)
            topk = 2 if num_experts > 2 else 1
            moe_module_transpose = Grouped_GEMM_MoE(
                hidden_size,
                expert_intermediate_size,
                0.0,
                num_experts,
                topk,
                use_swiglu=True,
                use_bias=False,
                use_expert_parallelism=False,
                merge_w1_v1=False,
                transpose_w1=True,
                implementation_type="Megatron",
                token_dispatcher_type="MindSpeedAllGather",
            ).to(device)
            moe_module_not_transpose = Grouped_GEMM_MoE(
                hidden_size,
                expert_intermediate_size,
                0.0,
                num_experts,
                topk,
                use_swiglu=True,
                use_bias=False,
                use_expert_parallelism=False,
                merge_w1_v1=False,
                transpose_w1=False,
                implementation_type="Megatron",
                token_dispatcher_type="MindSpeedAllGather",
            ).to(device)
            # sync parameters w1 -> (w1, v1), w2 ->w2
            with torch.no_grad():
                moe_module_not_transpose.w2.copy_(moe_module_transpose.w2)
                w1 = moe_module_transpose.w1.clone().transpose(-2, -1)
                v1 = moe_module_transpose.v1.clone().transpose(-2, -1)
                moe_module_not_transpose.w1.copy_(w1)
                moe_module_not_transpose.v1.copy_(v1)
            # create inputs
            inp = torch.randn(
                bs,
                seq_len,
                hidden_size,
                device=device,
                requires_grad=True,
            )
            expert_weights = torch.randn(
                bs,
                seq_len,
                topk,
                device=device,
            ).softmax(-1)

            top_experts = torch.randint(
                0,
                num_experts,
                (bs, seq_len, topk),
                device=device,
                dtype=torch.long,
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                merge_out = moe_module_transpose(inp, expert_weights, top_experts)
                not_merge_out = moe_module_not_transpose(inp, expert_weights, top_experts)
                merge_sum = torch.sum(merge_out)
                not_merge_sum = torch.sum(not_merge_out)
            merge_sum.backward()
            not_merge_sum.backward()
            self.assertTrue(torch.allclose(merge_out, not_merge_out, atol=1e-3))
            self.assertTrue(torch.allclose(moe_module_transpose.w2.grad, moe_module_not_transpose.w2.grad, atol=1e-2))

            # test no token
            inp = torch.empty([0, hidden_size], dtype=torch.float, device=device)
            tokens_per_expert = torch.zeros([num_experts], dtype=torch.int, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                res = moe_module_not_transpose.compute_expert(inp, tokens_per_expert)
            self.assertEqual(res.shape[0], 0)

    def test_grouped_gemm_moe_megatron(self):
        implementation_type = "Megatron"
        token_dispatcher_types = ["AllToAll", "AllGather", "MindSpeedAllGather", "MindSpeedAllToAll"]
        for num_experts, hidden_size, expert_intermediate_size, topk in [[8, 256, 1024, 2], [30, 512, 512, 6]]:
            for use_swiglu in [False, True]:
                for use_bias in [False, True]:
                    for token_dispatcher_type in token_dispatcher_types:
                        self._test_grouped_gemm_moe(
                            num_experts,
                            hidden_size,
                            expert_intermediate_size,
                            topk,
                            use_swiglu,
                            use_bias,
                            implementation_type,
                            token_dispatcher_type,
                        )

    def _test_grouped_gemm_moe(
        self,
        num_experts,
        hidden_size,
        expert_intermediate_size,
        topk,
        use_swiglu,
        use_bias,
        implementation_type="MegaBlocks",
        token_dispatcher_type="AllToAll",
    ):
        print(
            f"token_dispatcher_type {token_dispatcher_type}, num_experts {num_experts}, hidden_size  {hidden_size}, "
            f"expert_intermediate_size {expert_intermediate_size}, topk {topk}, "
            f"use_swiglu {use_swiglu}, use_bias {use_bias}"
        )
        GG_MoE = Grouped_GEMM_MoE(
            hidden_size,
            expert_intermediate_size,
            0.0,
            num_experts,
            topk,
            use_swiglu,
            use_bias,
            transpose_w1=False,
            implementation_type=implementation_type,
            token_dispatcher_type=token_dispatcher_type,
        )
        if implementation_type == "Megatron" and use_bias:
            use_bias = False
        ref_MoE = RefMoE(num_experts, hidden_size, expert_intermediate_size, use_swiglu, use_bias)
        fp32_ref_MoE = copy.deepcopy(ref_MoE)

        # align weight
        GG_MoE.load_state_dict(sd_trans(ref_MoE.state_dict(), num_experts, use_bias), strict=True)

        # to cuda
        fp32_ref_MoE.to(torch.cuda.current_device())
        ref_MoE.to(torch.cuda.current_device())
        GG_MoE.to(torch.cuda.current_device())

        # inputs, grad
        bs = 2
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
        assert_precision(fp32_ref_out, ref_out, GG_out, "output")
        ref_g = sd_trans({n: p.grad for n, p in ref_MoE.named_parameters()}, num_experts, use_bias)
        fp32_ref_g = sd_trans({n: p.grad for n, p in fp32_ref_MoE.named_parameters()}, num_experts, use_bias)
        for n, p in GG_MoE.named_parameters():
            assert_precision(fp32_ref_g[n], ref_g[n], p.grad, f"{n} grad")

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


def run_ep_grouped_gemm_moe(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    ep_mode = ([("expert", torch.distributed.get_world_size())], None)
    create_parallel_group(ep_mode)

    num_experts, hidden_size, expert_intermediate_size, topk = 4, 64, 128, 2
    for use_swiglu in [False, True]:
        for use_bias in [False, True]:
            for num_shared_experts in [0, 2]:
                _inter_run_ep_grouped_gemm_moe(
                    rank,
                    world_size,
                    num_experts,
                    hidden_size,
                    expert_intermediate_size,
                    topk,
                    use_swiglu,
                    use_bias,
                    num_shared_experts,
                )
    atorch.reset_distributed()


def run_ep_grouped_gemm_moe_single(rank, world_size, no_local_token=False, implementation_type="MegaBlocks"):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    ep_mode = ([("expert", torch.distributed.get_world_size())], None)
    create_parallel_group(ep_mode)

    num_experts, hidden_size, expert_intermediate_size, topk, num_shared_experts = 8, 64, 128, 2, 2
    use_swiglu, use_bias = False, False

    if implementation_type == "MegaBlocks":
        token_dispatcher_types = ["AllToAll"]
    elif implementation_type == "Megatron":
        token_dispatcher_types = ["AllToAll", "AllGather", "MindSpeedAllGather", "MindSpeedAllToAll"]

    for token_dispatcher_type in token_dispatcher_types:
        _inter_run_ep_grouped_gemm_moe(
            rank,
            world_size,
            num_experts,
            hidden_size,
            expert_intermediate_size,
            topk,
            use_swiglu,
            use_bias,
            num_shared_experts,
            no_local_token,
            implementation_type=implementation_type,
            token_dispatcher_type=token_dispatcher_type,
        )
    atorch.reset_distributed()


def _inter_run_ep_grouped_gemm_moe(
    rank,
    world_size,
    num_experts,
    hidden_size,
    expert_intermediate_size,
    topk,
    use_swiglu,
    use_bias,
    num_shared_experts,
    no_local_token=False,
    implementation_type="MegaBlocks",
    token_dispatcher_type="AllToAll",
):

    GG_MoE = Grouped_GEMM_MoE(
        hidden_size,
        expert_intermediate_size,
        0.0,
        num_experts,
        topk,
        use_swiglu,
        use_bias,
        use_expert_parallelism=True,
        transpose_w1=False,
        implementation_type=implementation_type,
        token_dispatcher_type=token_dispatcher_type,
    )
    # seed for same ref_MoE across rank
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)
    ref_MoE = RefMoE(num_experts, hidden_size, expert_intermediate_size, use_swiglu, use_bias, num_shared_experts)
    fp32_ref_MoE = copy.deepcopy(ref_MoE)

    if num_shared_experts > 0:
        GG_se = copy.deepcopy(ref_MoE.shared_experts)

        def se_fn1(x):
            x = GG_se.dense_h_to_4h(x)
            x = GG_se.activation_func(x)
            return x

        def se_fn2(x):
            x = GG_se.dense_4h_to_h(x)
            x = GG_se.dropout(x)
            return x

    # align weight
    ref_sd = ref_MoE.state_dict()

    # switch expert weight
    def ep_sd_trans(ref_sd, num_experts, use_bias):
        num_local_experts = divide(num_experts, world_size)
        for i in range(num_local_experts):
            src_idx = rank * num_local_experts + i
            ref_sd[f"experts.expert_{i}.dense_h_to_4h.weight"] = ref_sd[
                f"experts.expert_{src_idx}.dense_h_to_4h.weight"
            ]
            ref_sd[f"experts.expert_{i}.dense_4h_to_h.weight"] = ref_sd[
                f"experts.expert_{src_idx}.dense_4h_to_h.weight"
            ]
            if use_bias:
                ref_sd[f"experts.expert_{i}.dense_h_to_4h.bias"] = ref_sd[
                    f"experts.expert_{src_idx}.dense_h_to_4h.bias"
                ]
                ref_sd[f"experts.expert_{i}.dense_4h_to_h.bias"] = ref_sd[
                    f"experts.expert_{src_idx}.dense_4h_to_h.bias"
                ]
        return sd_trans(ref_sd, num_local_experts, use_bias)

    GG_MoE.load_state_dict(ep_sd_trans(ref_sd, num_experts, use_bias), strict=True)

    # to cuda
    fp32_ref_MoE.to(torch.cuda.current_device())
    ref_MoE.to(torch.cuda.current_device())
    GG_MoE.to(torch.cuda.current_device())
    if num_shared_experts > 0:
        GG_se.to(torch.cuda.current_device())

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

    if no_local_token:
        top_experts = torch.randint(
            0,
            num_experts // 2 - 1,
            (bs, seq_len, topk),
            device=torch.cuda.current_device(),
            dtype=torch.long,
        )
    else:
        top_experts = torch.randint(
            0,
            num_experts,
            (bs, seq_len, topk),
            device=torch.cuda.current_device(),
            dtype=torch.long,
        )
    rand_out_grad = torch.randn_like(inp)

    # fw bw
    rank_slice = (slice(rank, rank + 1), ...)
    se_fns = tuple() if num_shared_experts == 0 else (se_fn1, se_fn2)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        ref_out = ref_MoE(ref_inp, expert_weights, top_experts)
        GG_out = GG_MoE(inp[rank_slice], expert_weights[rank_slice], top_experts[rank_slice], *se_fns)
    fp32_ref_out = fp32_ref_MoE(fp32_ref_inp, expert_weights, top_experts)
    ref_out.backward(rand_out_grad)
    GG_out.backward(rand_out_grad[rank_slice])
    fp32_ref_out.backward(rand_out_grad)

    # check precision
    assert_precision(fp32_ref_out[rank_slice], ref_out[rank_slice], GG_out, "output")
    ref_g = ep_sd_trans({n: p.grad for n, p in ref_MoE.named_parameters()}, num_experts, use_bias)
    fp32_ref_g = ep_sd_trans({n: p.grad for n, p in fp32_ref_MoE.named_parameters()}, num_experts, use_bias)
    for n, p in GG_MoE.named_parameters():
        assert_precision(fp32_ref_g[n], ref_g[n], p.grad, f"{n} grad")

    if num_shared_experts > 0:
        fp32_ref_se_sd = {n: p.grad for n, p in fp32_ref_MoE.shared_experts.named_parameters()}
        ref_se_sd = {n: p.grad for n, p in ref_MoE.shared_experts.named_parameters()}
        for n, p in GG_se.named_parameters():
            g = p.grad
            dist.all_reduce(g)
            assert_precision(fp32_ref_se_sd[n], ref_se_sd[n], g)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() < (2, 0, 0),  # type: ignore
    "Must have at least 4 GPUs for expert parallel test",
)
class TestEPGroupedGemmMoE(unittest.TestCase):

    seed = 1234

    def setUp(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    @unittest.skipIf(is_torch_npu_available(), "MegaBlocks implementation_type not supported yet for npu")
    def test_ep_grouped_gemm_moe_megablocks(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ep_grouped_gemm_moe,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    def test_ep_grouped_gemm_moe_megatron(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ep_grouped_gemm_moe_single,
            args=(world_size, False, "Megatron"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(is_torch_npu_available(), "MegaBlocks implementation_type not supported yet for npu")
    def test_ep_grouped_gemm_moe_megablocks_no_local_token(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ep_grouped_gemm_moe_single,
            args=(world_size, True),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    def test_ep_grouped_gemm_moe_megatron_no_local_token(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ep_grouped_gemm_moe_single,
            args=(world_size, True, "Megatron"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(is_torch_npu_available(), "MegaBlocks implementation_type not supported yet for npu")
    def test_ep_grouped_gmm_moe_megablocks_prefetch_attn_and_expert(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["MOE_FSDP_PREFETCH_NUM"] = "2"
        mp.spawn(
            run_ep_grouped_gemm_moe_single,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
        del os.environ["MOE_FSDP_PREFETCH_NUM"]

    def test_ep_grouped_gmm_moe_megatron_prefetch_attn_and_expert(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["MOE_FSDP_PREFETCH_NUM"] = "2"
        mp.spawn(
            run_ep_grouped_gemm_moe_single,
            args=(world_size, False, "Megatron"),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
        del os.environ["MOE_FSDP_PREFETCH_NUM"]


class TranposeMOE(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        expert_intermediate_size,
        topk,
        use_swiglu,
        use_bias,
        implementation_type,
        token_dispatcher_type,
    ):
        super().__init__()
        self.moe1 = Grouped_GEMM_MoE(
            hidden_size,
            expert_intermediate_size,
            0.0,
            num_experts,
            topk,
            use_swiglu,
            use_bias,
            transpose_w1=False,
            implementation_type=implementation_type,
            token_dispatcher_type=token_dispatcher_type,
            merge_w1_v1=False,
        )
        self.moe2 = Grouped_GEMM_MoE(
            hidden_size,
            expert_intermediate_size,
            0.0,
            num_experts,
            topk,
            use_swiglu,
            use_bias,
            transpose_w1=False,
            implementation_type=implementation_type,
            token_dispatcher_type=token_dispatcher_type,
            merge_w1_v1=True,
        )
        self.m = torch.nn.Linear(2, 4)

    def forward(self, x):
        return x


def run_transpose_moe(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed(set_cuda_device_using_local_rank=True)
    num_experts, hidden_size, expert_intermediate_size, topk = 4, 2, 3, 2
    use_swiglu, use_bias = False, False
    implementation_type = "Megatron"
    token_dispatcher_type = "AllGather"
    model = TranposeMOE(
        hidden_size,
        expert_intermediate_size,
        num_experts,
        topk,
        use_swiglu,
        use_bias,
        implementation_type=implementation_type,
        token_dispatcher_type=token_dispatcher_type,
    )
    w1_v1_shape = (num_experts, hidden_size, expert_intermediate_size)

    temp = torch.arange(0, num_experts * hidden_size * expert_intermediate_size, dtype=torch.float32).reshape(
        w1_v1_shape
    )
    model.moe1.w1 = torch.nn.Parameter(temp)
    model.moe1.v1 = torch.nn.Parameter(temp.clone())
    model.moe2.w1 = torch.nn.Parameter(temp.clone())
    model.moe2.v1 = torch.nn.Parameter(temp.clone())
    model = FSDP(
        model, auto_wrap_policy=ModuleWrapPolicy({Grouped_GEMM_MoE}), sync_module_states=True, device_id=atorch.rank()
    )

    model = transpose_moe(model)

    target = temp.to("cuda").transpose(-1, -2).reshape(w1_v1_shape)
    with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(model, writeback=False):
        assert torch.allclose(model.moe1.w1, target)
        assert torch.allclose(model.moe1.v1, target)
        assert torch.allclose(model.moe2.w1, target)
        assert torch.allclose(model.moe2.v1, target)
    model(torch.ones((2, 2)).to("cuda"))

    atorch.reset_distributed()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() < (2, 0, 0),  # type: ignore
    "Must have at least 4 GPUs for expert parallel test",
)
class TestTransposeMOE(unittest.TestCase):
    def test_transpose_moe(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        mp.spawn(
            run_transpose_moe,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
