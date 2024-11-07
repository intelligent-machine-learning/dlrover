"""
Addapted from Megatron-LM with modification,
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/tokenizer/token_dispatcher.py
"""
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import torch

from atorch.common.env import EnvSetting
from atorch.communication.functions import (
    _gather_along_first_dim_expert_parallel,
    all_to_all,
    gather_from_expert_parallel_region_to_moe,
    reduce_scatter_to_expert_parallel_region_from_moe,
)
from atorch.distributed.distributed import parallel_group, parallel_group_size
from atorch.kernels import npu_fused_permute, npu_fused_unpermute
from atorch.utils.import_util import is_torch_npu_available


class NewIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, map_, value_, accumulate):
        ctx.map_ = map_
        ori_dtype = None
        if value_.dtype != torch.float32:
            ori_dtype = value_.dtype
            value_ = value_.float()
        output = tensor.index_put_(map_, value_, accumulate=accumulate)
        if ori_dtype:
            return output.to(ori_dtype)
        return output

    def backward(ctx, grad_output):
        map_ = ctx.map_
        return None, None, grad_output.index_select(0, map_[0]), None


def new_index_put(tensor, map_, value_, accumulate):
    return NewIndex.apply(tensor, map_, value_, accumulate)


class _MoeGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, map_):
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(input_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class _MoeScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        ctx.map = map_

        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=torch.cuda.current_device())
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def moe_gather(input_, map_):
    return _MoeGather.apply(input_, map_)


def moe_scatter(input_, map_, output_size=None):
    return _MoeScatter.apply(input_, map_, output_size)


def permute(tokens, indices, num_out_tokens: int = None):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor,
            should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor,
            should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    if is_torch_npu_available() and not EnvSetting().MOE_NPU_DISABLE_ARGSORT_REPLACE:
        # use float on npu for better perf
        sorted_indices = torch.argsort(flatten_indices.float())
    else:
        sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices,
    and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided,
            the unpermuted tokens will be merged with their respective probabilities.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    if is_torch_npu_available():
        # for better perf on npu
        unpermuted_tokens[sorted_indices] = permuted_tokens
    else:
        unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def npu_fused_unpermute_func(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
):
    # bfloat16 only for probs
    return npu_fused_unpermute(
        permuted_tokens,
        sorted_indices,
        probs.to(torch.bfloat16) if probs is not None else None,
        padded_mode=False,
        restore_shape=None,
    )


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        num_moe_experts: int,
        topk: int,
        add_bias: bool,
        use_expert_parallelism: bool,
    ) -> None:
        self.use_expert_parallelism = use_expert_parallelism
        self.num_local_experts = num_local_experts
        self.num_experts = num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        self.router_topk = topk
        self.add_bias = add_bias
        self.ep_size = parallel_group_size("expert") if parallel_group_size("expert") else 1

    @abstractmethod
    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        max_prob: torch.Tensor,
        max_ind: torch.Tensor,
        se_fn1: Optional[Callable] = None,
    ):
        """Dispatch tokens to experts.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            max_prob (torch.Tensor): Input tokens.
            max_ind (torch.Tensor): indices tensor.
            se_fn1 (Callable, optional): shared experts computation function1.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_input: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).
            se_fn2 (Callable, optional): shared experts computation function2.
            se_fn2_input (Tuple, optional): shared experts computation function2 input.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
                - Shared experts output.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEAllToAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        num_moe_experts: int,
        topk: int,
        add_bias: bool,
        use_expert_parallelism: bool,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(
            num_local_experts, local_expert_indices, num_moe_experts, topk, add_bias, use_expert_parallelism
        )

        self.use_fused_kernel = False
        self.unpermute_fn = unpermute

        self.hidden_shape: Optional[torch.Size] = None
        self.num_input_tokens = None
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert: Optional[torch.Tensor] = None
        self.num_out_tokens = None

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation.
        This method computes the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.ep_size
        self.num_out_tokens = num_local_tokens_per_expert.sum().cpu()

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"))
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(num_local_tokens_per_expert).reshape(
                ep_size, self.num_experts
            )
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices]
            self.output_splits = self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(-1, self.num_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(torch.device("cpu"), non_blocking=True)

        if self.num_local_experts > 1:
            expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        max_prob: torch.Tensor,
        max_ind: torch.Tensor,
        se_fn1: Optional[Callable] = None,
    ):
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            max_prob (torch.Tensor): Probs of tokens assigned to experts.
            max_ind (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = max_prob
        assert max_prob.dim() == 2, "Expected 2D tensor for probs"
        assert max_ind.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        se_intermediate = None
        tokens_per_expert = self.preprocess(max_ind)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = hidden_states.shape
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            max_ind,
            num_out_tokens=self.num_out_tokens,
        )

        if self.use_expert_parallelism:
            # Perform expert parallel AlltoAll communication
            global_input_tokens, git_handle = all_to_all(
                parallel_group("expert"),
                permutated_local_input_tokens,
                self.output_splits,
                self.input_splits,
                async_op=True,
            )
            if se_fn1 is not None and not EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                # Overlap with allgather
                se_intermediate = se_fn1(hidden_states)

            git_handle.wait()
            # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
            if self.num_local_experts > 1:
                global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
        else:
            global_input_tokens = permutated_local_input_tokens
            self.reversed_global_input_permutation_mapping = self.reversed_local_input_permutation_mapping

        return global_input_tokens, tokens_per_expert, se_intermediate

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_input: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
                - Shared experts output.
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        se_output = None

        if self.use_expert_parallelism:
            # Unpermutation 2: expert output to AlltoAll input
            if self.num_local_experts > 1:
                hidden_states = self.unpermute_fn(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping,
                )

            # Perform expert parallel AlltoAll communication
            # hidden_states: [SEQL, H] -> [SEQL, H/TP]
            permutated_local_input_tokens, plit_handle = all_to_all(
                parallel_group("expert"), hidden_states, self.input_splits, self.output_splits, async_op=True
            )
            if se_fn2 is not None and not EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                assert se_fn2_input is not None
                # Overlap with reduce_scatter
                se_output = se_fn2(*se_fn2_input)
            plit_handle.wait()
        else:
            permutated_local_input_tokens = hidden_states

        # Unpermutation 1: AlltoAll output to output
        output = self.unpermute_fn(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None, se_output


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        num_moe_experts: int,
        topk: int,
        add_bias: bool,
        use_expert_parallelism: bool,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(
            num_local_experts, local_expert_indices, num_moe_experts, topk, add_bias, use_expert_parallelism
        )

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs: Optional[torch.Tensor] = None

        # self.indices: The indices of `local_indices`
        # (which holds the un-sorted expert indices of tokens that local expert can process)
        # that give its sorted order along dim 0.
        self.indices: Optional[torch.Tensor] = None

        # self.global_local_map: 2D tensor.
        # A mask of mapping between global and local tokens
        # where each element is True if it's between the local_expert_indices.
        # Only useful when cross device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map: Optional[torch.Tensor] = None

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        max_prob: torch.Tensor,
        max_ind: torch.Tensor,
        se_fn1: Optional[Callable] = None,
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]
            max_prob: probs of local token assignment to global experts.
            max_ind: token assignment to local experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        se_intermediate = None

        # Permute the tokens across the expert parallel devices.
        if self.ep_size > 1:
            with torch.no_grad():
                global_indices = gather_from_expert_parallel_region_to_moe(max_ind)
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_mask)

            if self.router_topk > 1:  # k > 1
                global_probs = gather_from_expert_parallel_region_to_moe(max_prob)
                self.local_probs = global_probs.masked_select(global_local_mask)
            else:
                self.local_probs = max_prob

            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = gather_from_expert_parallel_region_to_moe(hidden_states, use_global_buffer=False)
            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = moe_gather(global_hidden_states, self.global_local_map)
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = max_ind
                self.local_probs = max_prob
                local_hidden_states = hidden_states
                self.global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            if is_torch_npu_available() and not EnvSetting().MOE_NPU_DISABLE_ARGSORT_REPLACE:
                # use float on npu for better perf
                self.indices = torch.argsort(local_indices.float(), dim=0)
            else:
                self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        if self.num_local_experts > 1:
            permuted_local_hidden_states = moe_gather(local_hidden_states, self.indices)
        else:
            permuted_local_hidden_states = local_hidden_states
        return (permuted_local_hidden_states, tokens_per_expert, se_intermediate)

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_input: Optional[Tuple] = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        assert bias is None, "Bias is not supported in MoEAllGatherlTokenDispatcher"

        # Stage1: unpermute the tokens and bias locally respectively.
        scores = self.local_probs.to(dtype=hidden_states.dtype)
        if self.num_local_experts > 1:
            assert self.indices.shape == hidden_states.shape
            unpermuted_local_hidden = moe_scatter(hidden_states, self.indices)
        else:
            unpermuted_local_hidden = hidden_states

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            raise NotImplementedError()

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        se_output = None
        if self.ep_size > 1:
            assert self.global_local_map is not None, "global_local_map is necessary for `AllGather`."
            ep_group_size = self.ep_size
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = moe_scatter(unpermuted_local_hidden, self.global_local_map, global_hidden_shape)
            output_total = reduce_scatter_to_expert_parallel_region_from_moe(unpermuted_global_hidden)
            if self.add_bias:
                raise NotImplementedError()
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(0, self.global_local_map, unpermuted_local_hidden)
                if self.add_bias:
                    raise NotImplementedError()

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            raise NotImplementedError()
        else:
            output_bias_total = None

        return output_total, output_bias_total, se_output


class MindSpeedAllGatherTokenDispatcher(MoEAllGatherTokenDispatcher):
    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        max_prob: torch.Tensor,
        max_ind: torch.Tensor,
        se_fn1: Optional[Callable] = None,
    ):
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        se_intermediate = None
        if self.ep_size > 1:
            # [S*B/TP, H] -> [S*B, H]
            with torch.no_grad():
                global_indices, gi_handle = (
                    max_ind
                    if isinstance(max_ind, tuple)
                    else gather_from_expert_parallel_region_to_moe(max_ind, async_op=True)
                )
            global_probs, gp_handle = gather_from_expert_parallel_region_to_moe(max_prob, async_op=True)
            global_hidden_states, ghs_handle = gather_from_expert_parallel_region_to_moe(hidden_states, async_op=True)
            if se_fn1 is not None and not EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                # Overlap with allgather
                se_intermediate = se_fn1(hidden_states)

            with torch.no_grad():
                gi_handle.wait()
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                if is_torch_npu_available():
                    local_indices = global_indices[global_local_mask]
                else:
                    local_indices = global_indices.masked_select(global_local_mask)

                if is_torch_npu_available() and not EnvSetting().MOE_NPU_DISABLE_ARGSORT_REPLACE:
                    # use float on npu for better perf
                    self.indices = torch.argsort(local_indices.float(), dim=0)
                else:
                    self.indices = torch.argsort(local_indices, dim=0)

                all_tokens_per_expert = torch.histc(
                    global_indices,
                    bins=self.num_local_experts * self.ep_size,
                    min=0,
                    max=self.num_local_experts * self.ep_size - 1,
                )
            self.all_tokens_per_expert = all_tokens_per_expert.cpu().to(torch.long)
            tokens_per_expert = self.all_tokens_per_expert[
                self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ]
            self.global_local_map = global_local_mask.nonzero()[:, 0]

            if self.router_topk > 1:  # k > 1
                gp_handle.wait()
                if is_torch_npu_available():
                    self.local_probs = global_probs[global_local_mask]
                else:
                    self.local_probs = global_probs.masked_select(global_local_mask)
            else:
                self.local_probs = max_prob

            ghs_handle.wait()
            local_hidden_states = global_hidden_states[self.global_local_map, :]
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                self.global_local_map = global_local_mask.nonzero()[:, 0]
                local_hidden_states = hidden_states[self.global_local_map, :]
            else:
                local_indices = max_ind.ravel()
                self.local_probs = max_prob
                local_hidden_states = hidden_states
                self.global_local_map = None

            with torch.no_grad():
                # The indices of local_indices that give its sorted order along dim 0.
                if is_torch_npu_available() and not EnvSetting().MOE_NPU_DISABLE_ARGSORT_REPLACE:
                    # use float on npu for better perf
                    self.indices = torch.argsort(local_indices.float(), dim=0)
                else:
                    self.indices = torch.argsort(local_indices, dim=0)
                tokens_per_expert = torch.histc(
                    local_indices,
                    bins=self.num_local_experts,
                    min=self.local_expert_indices[0],
                    max=self.local_expert_indices[-1],
                )
                tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
            self.all_tokens_per_expert = tokens_per_expert

        permuted_local_hidden_states = local_hidden_states[self.indices, :]
        return (permuted_local_hidden_states, tokens_per_expert, se_intermediate)

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_input: Optional[Tuple] = None,
    ):
        # Stage1: unpermute the tokens and bias locally respectively.w
        scores = self.local_probs.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        unpermuted_local_hidden.index_put_((self.indices,), hidden_states[: self.indices.shape[0], :], accumulate=False)

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            raise NotImplementedError()

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        se_output = None
        if self.ep_size > 1:
            assert self.global_local_map is not None, "global_local_map is necessary for `AllGather`."
            ep_group_size = self.ep_size
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            if is_torch_npu_available() and EnvSetting().MOE_REPLACE_MINDSPEED_ALLGATHER_TOKENDISPATCHER_INDEX:
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape, dtype=torch.float, device=torch.cuda.current_device()
                )
                unpermuted_global_hidden = new_index_put(
                    unpermuted_global_hidden,
                    (self.global_local_map,),
                    unpermuted_local_hidden[: self.global_local_map.shape[0], :],
                    accumulate=True,
                )
            else:
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device()
                )
                unpermuted_global_hidden.index_put_(
                    (self.global_local_map,),
                    unpermuted_local_hidden[: self.global_local_map.shape[0], :],
                    accumulate=True,
                )

            output_total, ot_handle = reduce_scatter_to_expert_parallel_region_from_moe(
                unpermuted_global_hidden, async_op=True
            )
            if se_fn2 is not None and not EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                assert se_fn2_input is not None
                # Overlap with reduce_scatter
                se_output = se_fn2(*se_fn2_input)

            if self.add_bias:
                raise NotImplementedError()
            ot_handle.wait()
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.index_put(
                    (self.global_local_map,),
                    unpermuted_local_hidden[: self.global_local_map.shape[0], :],
                    accumulate=True,
                )
                if self.add_bias:
                    raise NotImplementedError()

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            raise NotImplementedError()
        else:
            output_bias_total = None

        return output_total, output_bias_total, se_output


class MindSpeedAllToAllTokenDispatcher(MoEAllToAllTokenDispatcher):
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        num_moe_experts: int,
        topk: int,
        add_bias: bool,
        use_expert_parallelism: bool,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher. use fused ops if available and not disabled.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(
            num_local_experts, local_expert_indices, num_moe_experts, topk, add_bias, use_expert_parallelism
        )
        if (
            is_torch_npu_available()
            and not EnvSetting().MOE_NPU_DISABLE_FUSED_KERNEL
            and npu_fused_permute is not None
            and npu_fused_unpermute is not None
        ):
            self.use_fused_kernel = True
            self.unpermute_fn = npu_fused_unpermute_func
        else:
            self.use_fused_kernel = False

    def _permute(self, tokens, indices, topk: int = 1, num_out_tokens: int = 0):
        if self.use_fused_kernel:
            return npu_fused_permute(
                tokens, indices.view(-1, topk).to(torch.int64), num_out_tokens=num_out_tokens, padded_mode=False
            )

        if topk > 1:
            assert indices.size(1) == topk
        flatten_indices = indices.view(-1)
        # sorted_indices = torch.argsort(flatten_indices, stable=True)  # argsort int64 will be run on host cpu
        sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.ep_size
        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"))
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(num_local_tokens_per_expert).reshape(
                ep_size, self.num_experts
            )
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices]
            self.output_splits = self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(-1, self.num_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(torch.device("cpu"), non_blocking=True)

        if self.num_local_experts > 1:
            if not hasattr(self, "comm_stream"):
                self.comm_stream = torch.cuda.Stream()
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                expert_ids_per_ep_rank = torch.tensor(
                    [i % self.num_local_experts for i in range(self.num_experts)],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                    expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
                )

        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        max_prob: torch.Tensor,
        max_ind: torch.Tensor,
        se_fn1: Optional[Callable] = None,
    ):
        self.hidden_shape = hidden_states.shape
        # self.scores = scores
        self.probs = max_prob

        assert max_prob.dim() == 2, "Expected 2D tensor for scores"
        assert max_ind.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(max_ind)

        # Flatten the input tensor
        # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        se_intermediate = None

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        # skip

        # Permutation 1: input to AlltoAll input
        self.local_input_tokens_global_experts_indices = max_ind
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = self._permute(
            hidden_states,
            self.local_input_tokens_global_experts_indices,
            topk=self.router_topk,
        )

        if self.use_expert_parallelism:
            # Perform expert parallel AlltoAll communication
            global_input_tokens, git_handle = all_to_all(
                parallel_group("expert"),
                permutated_local_input_tokens,
                self.output_splits,
                self.input_splits,
                async_op=True,
            )

            if se_fn1 is not None and not EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                # Overlap with AlltoAll
                se_intermediate = se_fn1(hidden_states)

            git_handle.wait()
            # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
            if self.num_local_experts > 1:
                torch.cuda.current_stream().wait_stream(self.comm_stream)
                global_input_tokens, self.reversed_global_input_permutation_mapping = self._permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
        else:
            global_input_tokens = permutated_local_input_tokens
            self.reversed_global_input_permutation_mapping = self.reversed_local_input_permutation_mapping

        # Perform tensor parallel All-Gather
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        # skip

        return global_input_tokens, tokens_per_expert, se_intermediate
