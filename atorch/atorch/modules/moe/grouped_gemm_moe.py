try:
    import megablocks.ops as ops
except (ImportError, RuntimeError):
    ops = None

import os
from typing import Callable, List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from torch.distributed.fsdp import _common_utils, _runtime_utils
except (ImportError, ModuleNotFoundError):
    _runtime_utils = None
    _common_utils = None


from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import divide
from atorch.distributed.distributed import parallel_group
from atorch.kernels import bias_gather_add, gmm
from atorch.utils.version import torch_version

_MOE_ALL2ALL_OVERLAP_BACKWARD = False
_OLD_POST_BACKWARD_HOOK = None
_OLD_PREFETCH_HANDLE = None

# TODO: config the num in moe module which is more comprehensive
_MOE_FSDP_PREFETCH_NUM = int(os.getenv("MOE_FSDP_PREFETCH_NUM", 1))


def patch_fsdp_post_backward_hook():
    global _OLD_POST_BACKWARD_HOOK
    if _OLD_POST_BACKWARD_HOOK is not None:
        return

    if _runtime_utils is None:
        logger.warning("Can't import fsdp runtime utils correctly, skip fsdp `post_backward_hook` patching for moe.")
        return

    @no_type_check
    @torch.no_grad()
    def _atorch_post_backward_hook(
        state,
        handle,
        flat_param,
        *unused,
    ):
        global _MOE_ALL2ALL_OVERLAP_BACKWARD
        if _MOE_ALL2ALL_OVERLAP_BACKWARD:
            _MOE_ALL2ALL_OVERLAP_BACKWARD = False
            return

        return _OLD_POST_BACKWARD_HOOK(state, handle, flat_param, *unused)

    _OLD_POST_BACKWARD_HOOK = _runtime_utils._post_backward_hook
    _runtime_utils._post_backward_hook = _atorch_post_backward_hook


def patch_fsdp_prefetch_handle():
    global _OLD_PREFETCH_HANDLE
    if _OLD_PREFETCH_HANDLE is not None:
        return

    if _runtime_utils is None:
        logger.warning("Can't import fsdp runtime utils correctly, skip fsdp `prefetch_handle` patching for moe.")
        return

    @no_type_check
    def _atorch_prefetch_handle(state, current_handle, prefetch_mode):
        """
        Prefetches the next handles if needed (without synchronization). An empty
        handles key cannot prefetch.
        """
        if not current_handle:
            return None
        handle = _runtime_utils._get_handle_to_prefetch(state, current_handle)
        if not handle:
            return None
        # Temporarily emulate the training state while calling `_unshard` to
        # ensure the correct `as_params` for `_use_unsharded_views()`
        prev_training_state = handle._training_state
        if prefetch_mode == _runtime_utils._PrefetchMode.BACKWARD:
            handle._training_state = _common_utils.HandleTrainingState.BACKWARD_PRE
        elif prefetch_mode == _runtime_utils._PrefetchMode.FORWARD:
            handle._training_state = _common_utils.HandleTrainingState.FORWARD
        else:
            raise ValueError(f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}")
        # Prefetch the next set of handles without synchronizing to allow
        # the sync to happen as late as possible to maximize overlap
        _runtime_utils._unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
        handle._training_state = prev_training_state
        handle._prefetched = True

        return handle

    @no_type_check
    def _atorch_prefetch_handle_wrapper(state, current_handle, prefetch_mode):
        global _MOE_FSDP_PREFETCH_NUM
        next_handle = current_handle
        for _ in range(_MOE_FSDP_PREFETCH_NUM):
            prev_training_state = next_handle._training_state if next_handle else None
            next_handle = _atorch_prefetch_handle(state, current_handle=next_handle, prefetch_mode=prefetch_mode)
            if next_handle:
                next_handle._training_state = prev_training_state

    _OLD_PREFETCH_HANDLE = _runtime_utils._prefetch_handle
    _runtime_utils._prefetch_handle = _atorch_prefetch_handle_wrapper


if torch_version() >= (2, 1, 0):  # type: ignore
    patch_fsdp_post_backward_hook()
    if _MOE_FSDP_PREFETCH_NUM > 1:
        patch_fsdp_prefetch_handle()
else:
    logger.warning(
        "Can't patch torch {} FSDP runtime util `should_free_in_backward` now, \
            we skip it, please update to version 2.1.0 at least.".format(
            torch_version()
        )
    )


# refer to megablocks
class AllToAllWithComputeOverlapOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op, compute_fn, *compute_fn_input):
        out = torch.empty((sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = torch.distributed.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )  # async for compute overlap

        # fw compute
        detached_compute_out = None
        if compute_fn is not None:
            detached_compute_fn_input = [inp.detach().requires_grad_() for inp in compute_fn_input]
            with torch.enable_grad():
                compute_out = compute_fn(*detached_compute_fn_input)

            ctx.save_for_backward(compute_out, *detached_compute_fn_input)
            detached_compute_out = compute_out.detach().requires_grad_()

        if async_op:
            return out, handle, detached_compute_out
        else:
            handle.wait()
            return out, None, detached_compute_out

    @staticmethod
    def backward(ctx, grad, _, compute_grad):
        out, handle = None, None
        if ctx.needs_input_grad[0]:
            out = torch.empty(ctx.input_shape, device=grad.device, dtype=grad.dtype)
            handle = torch.distributed.all_to_all_single(
                out,
                grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
                async_op=True,
            )  # async for compute overlap

        # bw compute
        if compute_grad is not None:
            saved_tensors = ctx.saved_tensors
            compute_out, detached_compute_fn_input = saved_tensors[0], saved_tensors[1:]
            if compute_out is not None:
                global _MOE_ALL2ALL_OVERLAP_BACKWARD
                _MOE_ALL2ALL_OVERLAP_BACKWARD = True
                torch.autograd.backward((compute_out,), (compute_grad,))

        if handle is not None:
            handle.wait()

        if compute_grad is not None:
            detached_compute_fn_input_grad = tuple([detached_inp.grad for detached_inp in detached_compute_fn_input])
            return (
                out,
                None,
                None,
                None,
                None,
                None,
            ) + detached_compute_fn_input_grad
        else:
            return (out, None, None, None, None, None, None)


def all_to_all_with_compute_overlap(
    x, output_split_sizes, input_split_sizes, group, async_op, compute_fn, *compute_fn_input
):
    """
    insert compute bw/fw in a2a async. Currently only support single tensor
    returned from compute_fn.
    Warning:: make sure the bounded tensors in compute_fn are all parameters
    """
    return AllToAllWithComputeOverlapOp.apply(
        x, output_split_sizes, input_split_sizes, group, async_op, compute_fn, *compute_fn_input
    )


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
        use_expert_parallelism=False,
        expert_parallel_group=None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.output_dropout_prob = output_dropout_prob
        self.num_experts = num_experts
        self.use_expert_parallelism = use_expert_parallelism
        if self.use_expert_parallelism:
            if expert_parallel_group is None:
                self.expert_parallel_group = parallel_group("expert")
            else:
                self.expert_parallel_group = expert_parallel_group
        self.num_local_experts = divide(
            self.num_experts, dist.get_world_size(self.expert_parallel_group) if self.use_expert_parallelism else 1
        )
        self.topk = topk
        self.use_swiglu = use_swiglu
        self.use_bias = use_bias

        # rm `fine_grained_factor`, make sure num_experts, top_k have been `update_moe_config`
        # and pass the updated `expert_intermediate_size`

        self.activation = SwiGLUActivatition() if self.use_swiglu else torch.nn.functional.gelu

        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_local_experts, self.hidden_size, self.expert_intermediate_size * (2 if self.use_swiglu else 1)
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(self.num_local_experts, self.expert_intermediate_size, self.hidden_size)
        )
        if self.use_bias:
            self.b1 = torch.nn.Parameter(
                torch.empty(self.num_local_experts, self.expert_intermediate_size * (2 if self.use_swiglu else 1))
            )
            self.b2 = torch.nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size))

        self.dropout = torch.nn.Dropout(self.output_dropout_prob)

        # reset gg weight bias
        self.w1.data.normal_(mean=0.0, std=initializer_range)
        self.w2.data.normal_(mean=0.0, std=initializer_range)
        if self.use_bias:
            self.b1.data.zero_()
            self.b2.data.zero_()

        # megablocks gather scatter
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.local_sort_end_bit = max(int(np.ceil(np.log2(self.num_local_experts))), 1)

    @torch.no_grad()
    def indices_and_bins(self, top_experts):
        """refer to megablocks"""
        top_expert = top_experts.flatten().int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    @torch.no_grad()
    def local_indices_and_bins(self, parallel_tokens_per_expert, tokens_received):
        """refer to megablocks, parallel_forward_once"""
        replicate_bins = ops.inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)
        replicate_bins = replicate_bins.view(1) if not len(replicate_bins.size()) else replicate_bins

        # Construct the expert indices for the permuted tokens.
        parallel_top_expert = torch.remainder(
            torch.arange(self.num_experts, dtype=torch.int32, device=parallel_tokens_per_expert.device),
            self.num_local_experts,
        )
        parallel_top_expert = ops.replicate(
            parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
        ).flatten()

        parallel_bin_ids, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

        # Calculate the bins boundaries from the token counts.
        parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
        parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
        parallel_bins = parallel_bins.view(1) if not len(parallel_bins.size()) else parallel_bins
        return parallel_indices, parallel_bin_ids, parallel_bins, parallel_tokens_per_expert

    @torch.no_grad()
    def get_send_recv_counts(self, tpe_handle, tokens_per_expert, parallel_tokens_per_expert):
        """refer to megablocks, parallel_forward_once"""
        tpe_handle.wait()

        # Reshape to [ep_world_size, self.num_local_experts].
        ep_world_size = torch.distributed.get_world_size(self.expert_parallel_group)
        tokens_per_expert = tokens_per_expert.view(ep_world_size, self.num_local_experts)
        parallel_tokens_per_expert = parallel_tokens_per_expert.view(ep_world_size, self.num_local_experts)

        send_counts = tokens_per_expert.cpu().sum(dim=-1)
        parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
        recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

        # Convert the send/recv counts to lists.
        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()
        tokens_received = sum(recv_counts)
        return send_counts, recv_counts, tokens_received, parallel_tokens_per_expert

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        top_experts: torch.Tensor,
        se_fn1: Optional[Callable] = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_additional_input: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ):
        """
        se_fn: shared expert compute function 1 or 2
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        expert_weights = expert_weights.flatten()
        indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(top_experts)

        # pre launch a2a tokens_per_expert
        if self.use_expert_parallelism:
            parallel_tokens_per_expert = torch.empty_like(tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert, tokens_per_expert, group=self.expert_parallel_group, async_op=True
            )

        # permute hidden states
        permuted_hidden_states = ops.gather(hidden_states, indices, bin_ids, bins, self.topk)

        # get recv/send counts and a2a permuted hidden states
        if self.use_expert_parallelism:
            send_counts, recv_counts, tokens_received, parallel_tokens_per_expert = self.get_send_recv_counts(
                tpe_handle, tokens_per_expert, parallel_tokens_per_expert
            )
            A2Aed_hidden_states, a2a_hs_handle, se_intermediate = all_to_all_with_compute_overlap(
                permuted_hidden_states,
                recv_counts,
                send_counts,
                self.expert_parallel_group,
                True,
                se_fn1,
                hidden_states if se_fn1 is not None else None,
            )
            # local permute All2Alled hidden states if num_local_experts > 1
            if self.num_local_experts > 1:
                (
                    parallel_indices,
                    parallel_bin_ids,
                    parallel_bins,
                    parallel_tokens_per_expert,
                ) = self.local_indices_and_bins(parallel_tokens_per_expert, tokens_received)
                a2a_hs_handle.wait()
                _hidden_states = ops.gather(A2Aed_hidden_states, parallel_indices, parallel_bin_ids, parallel_bins, 1)
                _bin_ids = parallel_bin_ids
            else:
                a2a_hs_handle.wait()
                _hidden_states = A2Aed_hidden_states
            _tokens_per_expert = parallel_tokens_per_expert
        else:
            _hidden_states = permuted_hidden_states
            if se_fn1 is not None:
                se_intermediate = se_fn1(hidden_states)

            _tokens_per_expert = tokens_per_expert
            _bin_ids = bin_ids

        # compute
        if _hidden_states.nelement() == 0:
            # no valid token per expert & permuted local hidden states
            assert _tokens_per_expert.sum(dim=0) == 0
            expert_output = self.compute_expert_no_token(_hidden_states)
        else:
            if self.num_local_experts > 1:
                expert_output = self.compute_moe(_hidden_states, _tokens_per_expert, _bin_ids)
            else:
                expert_output = self.compute_single_expert(_hidden_states)

        # ep post compute scatter/a2a
        se_output = None
        if se_fn2_additional_input is not None:
            if isinstance(se_fn2_additional_input, (tuple, list)):
                compute_fn_input = (se_intermediate,) + tuple(se_fn2_additional_input)
            else:
                compute_fn_input = (se_intermediate, se_fn2_additional_input)
        elif se_fn2 is not None:
            compute_fn_input = (se_intermediate,)
        if self.use_expert_parallelism:
            if self.num_local_experts > 1:
                # scatter back
                expert_output = ops.scatter(expert_output, parallel_indices, parallel_bin_ids, None, parallel_bins, 1)
            # a2a back expert_output
            expert_output, _, se_output = all_to_all_with_compute_overlap(
                expert_output,
                send_counts,
                recv_counts,
                self.expert_parallel_group,
                False,
                se_fn2,
                *compute_fn_input if se_fn2 is not None else (None,),
            )
        else:
            if se_fn2 is not None:
                se_output = se_fn2(*compute_fn_input)

        # unpermute expert outpute
        output = ops.scatter(expert_output, indices, bin_ids, expert_weights, bins, self.topk).view(self.hidden_shape)
        if se_output is not None:
            output += se_output.view(self.hidden_shape)

        return output

    def compute_moe(self, hidden_states, tokens_per_expert, bin_ids):
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
        fc1_output = gmm(hidden_states, self.w1, tokens_per_expert, trans_b=False)
        if self.use_bias:
            fc1_output = bias_gather_add(fc1_output, self.b1, bin_ids)
        intermediate_states = self.activation(fc1_output)
        fc2_output = gmm(intermediate_states, self.w2, tokens_per_expert, trans_b=False)
        if self.use_bias:
            fc2_output = bias_gather_add(fc2_output, self.b2, bin_ids)
        expert_output = self.dropout(fc2_output)
        return expert_output

    def compute_single_expert(self, hidden_states):
        fc1_output = F.linear(
            hidden_states, self.w1.squeeze(0).transpose(0, 1), self.b1.squeeze(0) if self.use_bias else None
        )
        intermediate_states = self.activation(fc1_output)
        fc2_output = F.linear(
            intermediate_states, self.w2.squeeze(0).transpose(0, 1), self.b2.squeeze(0) if self.use_bias else None
        )
        expert_output = self.dropout(fc2_output)
        return expert_output

    def compute_expert_no_token(self, hidden_states):
        w1 = self.w1.view(self.hidden_size, -1)
        w2 = self.w2.view(-1, self.hidden_size)
        fc1_output = torch.matmul(hidden_states, w1)
        if self.use_bias:
            raise NotImplementedError("Not support bias when there is no local token.")
        intermediate_states = self.activation(fc1_output)
        fc2_output = torch.matmul(intermediate_states, w2)
        return fc2_output
