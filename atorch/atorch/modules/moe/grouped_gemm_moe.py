try:
    import megablocks.ops as ops
except (ImportError, RuntimeError):
    ops = None

from enum import Enum
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


try:
    from torch.utils import checkpoint
except (ImportError, ModuleNotFoundError):
    checkpoint = None


from atorch.common.env import EnvSetting
from atorch.common.log_utils import default_logger as logger
from atorch.common.singleton import SingletonMeta
from atorch.common.util_func import divide
from atorch.distributed.distributed import parallel_group, parallel_group_size, parallel_rank, rank
from atorch.kernels import bias_gather_add
from atorch.modules.moe.token_dispatcher import (
    MindSpeedAllGatherTokenDispatcher,
    MindSpeedAllToAllTokenDispatcher,
    MoEAllGatherTokenDispatcher,
    MoEAllToAllTokenDispatcher,
)
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.version import torch_version

if is_torch_npu_available():
    from atorch.npu.gmm import npu_gmm as gmm
else:
    from atorch.kernels import gmm


class MOEContext(metaclass=SingletonMeta):
    MOE_ALL2ALL_OVERLAP_BACKWARD = False
    MOE_ALL2ALL_OVERLAP_BACKWARD_FOR_CHECKPOINT = False
    OLD_POST_BACKWARD_HOOK = None
    OLD_PREFETCH_HANDLE = None
    OLD_CHECKPOINT_HOOK = None


def patch_fsdp_post_backward_hook():
    if MOEContext().OLD_POST_BACKWARD_HOOK is not None:
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
        if MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD:
            MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD = False
            return

        return MOEContext().OLD_POST_BACKWARD_HOOK(state, handle, flat_param, *unused)

    MOEContext().OLD_POST_BACKWARD_HOOK = _runtime_utils._post_backward_hook
    _runtime_utils._post_backward_hook = _atorch_post_backward_hook


def patch_fsdp_prefetch_handle():
    if MOEContext().OLD_PREFETCH_HANDLE is not None:
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
        next_handle = current_handle
        for _ in range(EnvSetting().MOE_FSDP_PREFETCH_NUM):
            prev_training_state = next_handle._training_state if next_handle else None
            next_handle = _atorch_prefetch_handle(state, current_handle=next_handle, prefetch_mode=prefetch_mode)
            if next_handle and EnvSetting().MOE_FSDP_PREFETCH_NUM > 1:
                next_handle._training_state = prev_training_state

    MOEContext().OLD_PREFETCH_HANDLE = _runtime_utils._prefetch_handle
    _runtime_utils._prefetch_handle = _atorch_prefetch_handle_wrapper


def patch_checkpoint_hook():
    if MOEContext().OLD_CHECKPOINT_HOOK is not None:
        return

    if checkpoint is None:
        logger.warning("Can't import torch checkpoint correctly, skip checkpoint `_checkpoint_hook` patching for moe.")
        return

    import uuid
    import weakref

    class _atorch_checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
        def __init__(self, frame):
            def pack_hook(x):
                # See Rule 4 above
                holder = checkpoint._Holder()
                frame.weak_holders.append(weakref.ref(holder))
                # Save metadata to detect non-determinism
                if frame.metadata_fn is not None:
                    with torch.no_grad():
                        frame.x_metadatas.append(frame.metadata_fn(x))
                return holder

            def unpack_hook(holder):
                if MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD_FOR_CHECKPOINT:
                    for id in holder.handles:
                        if holder.handles[id] is not None and holder.handles[id] in frame.recomputed[id]:
                            return frame.recomputed[id][holder.handles[id]]

                gid = torch._C._current_graph_task_id()
                if gid == -1:
                    # generate a temporary id if we trigger unpack outside of a backward call
                    gid = int(uuid.uuid4())

                if not frame.is_recomputed[gid]:
                    ctx = frame.input_saver.grad_fn
                    args = ctx.get_args(ctx.saved_tensors)

                    try:
                        with checkpoint._recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
                            frame.recompute_fn(*args)
                    except checkpoint._StopRecomputationError:
                        pass
                    frame.is_recomputed[gid] = True
                    frame.check_recomputed_tensors_match(gid)

                checkpoint._internal_assert(gid in holder.handles)

                if holder.handles[gid] is None:
                    raise checkpoint.CheckpointError(
                        "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                        "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                        "so only once. Otherwise please open an issue with details on your use case."
                    )
                checkpoint._internal_assert(holder.handles[gid] in frame.recomputed[gid])
                ret = frame.recomputed[gid][holder.handles[gid]]
                holder.handles[gid] = None
                return ret

            if frame.unpack_error_cb is not None:

                def unpack_hook_with_error_cb(holder):
                    try:
                        return unpack_hook(holder)
                    except checkpoint.CheckpointError as e:
                        frame.unpack_error_cb(e)

                super().__init__(pack_hook, unpack_hook_with_error_cb)
            else:
                super().__init__(pack_hook, unpack_hook)

    MOEContext().OLD_CHECKPOINT_HOOK = checkpoint._checkpoint_hook
    checkpoint._checkpoint_hook = _atorch_checkpoint_hook


if torch_version() >= (2, 1, 0):  # type: ignore
    patch_fsdp_post_backward_hook()
    if not EnvSetting().DISABLE_CHECKPOINT_PATCH:
        patch_checkpoint_hook()
    if EnvSetting().MOE_FSDP_PREFETCH_NUM > 1:
        patch_fsdp_prefetch_handle()
else:
    logger.warning(
        "Can't patch torch {} FSDP runtime util `should_free_in_backward` now, \
            we skip it, please update to version 2.1.0 at least.".format(
            torch_version()
        )
    )


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


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
            MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD_FOR_CHECKPOINT = True
            saved_tensors = ctx.saved_tensors
            compute_out, detached_compute_fn_input = saved_tensors[0], saved_tensors[1:]
            if compute_out is not None:
                MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD = True
                torch.autograd.backward((compute_out,), (compute_grad,))

        if handle is not None:
            handle.wait()

        if compute_grad is not None:
            MOEContext().MOE_ALL2ALL_OVERLAP_BACKWARD_FOR_CHECKPOINT = False
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


class MOEImplmenterType(Enum):
    MegaBlocks = "MegaBlocks"
    Megatron = "Megatron"


class MOETokenDispatcherType(Enum):
    AllGather = "AllGather"
    MindSpeedAllGather = "MindSpeedAllGather"
    AllToAll = "AllToAll"
    MindSpeedAllToAll = "MindSpeedAllToAll"


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
        use_bias=False,
        initializer_range=0.02,
        use_expert_parallelism=False,
        expert_parallel_group=None,
        merge_w1_v1=True,
        transpose_w1=True,
        is_scale_gradient=False,
        implementation_type="MegaBlocks",
        token_dispatcher_type="AllToAll",
    ) -> None:
        super().__init__()

        self.mlp_prefix = EnvSetting().MOE_MLP_PREFIX
        self.implementer_type = MOEImplmenterType(implementation_type)
        self.token_dispatcher_type = MOETokenDispatcherType(token_dispatcher_type)
        if is_torch_npu_available():
            if self.implementer_type is not MOEImplmenterType.Megatron:
                self.implementer_type = MOEImplmenterType.Megatron
                if rank() is None or rank() == 0:
                    logger.warning(
                        "MegaBlocks implementation_type not supported yet for npu,\
                            using Megatron implementation type instead"
                    )
        if self.implementer_type is MOEImplmenterType.Megatron and use_bias:
            use_bias = False
            if rank() is None or rank() == 0:
                logger.warning("Grouped_GEMM_MoE Megatron implementation not support use_bias.")

        if self.implementer_type is MOEImplmenterType.MegaBlocks and (
            self.token_dispatcher_type is MOETokenDispatcherType.AllGather
            or self.token_dispatcher_type is MOETokenDispatcherType.MindSpeedAllGather
            or self.token_dispatcher_type is MOETokenDispatcherType.MindSpeedAllToAll
        ):
            self.token_dispatcher_type = MOETokenDispatcherType.AllToAll
            if rank() is None or rank() == 0:
                logger.warning(
                    "MegaBlocks implementation_type is suitable for AllToAll token dispatcher type only,\
                        using AllToAll token dispatcher type instead"
                )

        if (
            self.implementer_type is MOEImplmenterType.Megatron
            and self.token_dispatcher_type is not MOETokenDispatcherType.MindSpeedAllGather
            and EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP
        ):
            if rank() is None or rank() == 0:
                logger.warning(
                    f"Don't support shared expert overlap for MOEImplmenterType.Megatron \
                        token_dispatcher_type {self.token_dispatcher_type} now, \
                        we ignore the MOE_DISABLE_SHARED_EXPERT_OVERLAP env."
                )

        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.output_dropout_prob = output_dropout_prob
        self.num_experts = num_experts
        self.use_expert_parallelism = use_expert_parallelism
        self.is_scale_gradient = is_scale_gradient
        self.gradient_scale = None
        if self.use_expert_parallelism and self.is_scale_gradient:
            self.gradient_scale = 1 / parallel_group_size("expert")
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
        self.merge_w1_v1 = merge_w1_v1
        self.transpose_w1 = transpose_w1

        if self.mlp_prefix:
            self.mlp = torch.nn.ParameterDict({})

        # rm `fine_grained_factor`, make sure num_experts, top_k have been `update_moe_config`
        # and pass the updated `expert_intermediate_size`

        self.activation = SwiGLUActivatition() if self.use_swiglu else torch.nn.functional.gelu

        if self.use_swiglu and not self.merge_w1_v1:
            if self.transpose_w1:
                w1_v1_shape = (self.num_local_experts, self.expert_intermediate_size, self.hidden_size)
            else:
                w1_v1_shape = (self.num_local_experts, self.hidden_size, self.expert_intermediate_size)

            w1_param = torch.nn.Parameter(torch.empty(w1_v1_shape))
            w2_param = torch.nn.Parameter(
                torch.empty(self.num_local_experts, self.expert_intermediate_size, self.hidden_size)
            )
            v1_param = torch.nn.Parameter(torch.empty(w1_v1_shape))

            if self.mlp_prefix:
                self.mlp["w1"] = w1_param
                self.mlp["w2"] = w2_param
                self.mlp["v1"] = v1_param
            else:
                self.w1 = w1_param
                self.w2 = w2_param
                self.v1 = v1_param
            assert not self.use_bias, "bias not supported yet for not merge_w1_v1"
        else:
            if self.transpose_w1:
                w1_shape = (
                    self.num_local_experts,
                    self.expert_intermediate_size * (2 if self.use_swiglu else 1),
                    self.hidden_size,
                )
            else:
                w1_shape = (
                    self.num_local_experts,
                    self.hidden_size,
                    self.expert_intermediate_size * (2 if self.use_swiglu else 1),
                )

            w1_param = torch.nn.Parameter(torch.empty(w1_shape))
            w2_param = torch.nn.Parameter(
                torch.empty(self.num_local_experts, self.expert_intermediate_size, self.hidden_size)
            )

            if self.mlp_prefix:
                self.mlp["w1"] = w1_param
                self.mlp["w2"] = w2_param
            else:
                self.w1 = w1_param
                self.w2 = w2_param
        if self.use_bias:
            # TODO: support mlp prefix for b1/b2
            self.b1 = torch.nn.Parameter(
                torch.empty(self.num_local_experts, self.expert_intermediate_size * (2 if self.use_swiglu else 1))
            )
            self.b2 = torch.nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size))

        self.dropout = torch.nn.Dropout(self.output_dropout_prob)

        # reset gg weight bias
        self.w1.data.normal_(mean=0.0, std=initializer_range)
        self.w2.data.normal_(mean=0.0, std=initializer_range)
        if self.has_v1():
            self.v1.data.normal_(mean=0.0, std=initializer_range)
        if self.use_bias:
            self.b1.data.zero_()
            self.b2.data.zero_()

        if self.implementer_type is MOEImplmenterType.Megatron:
            self._additional_init_for_megatron_moe()
        elif self.implementer_type is MOEImplmenterType.MegaBlocks:
            self._additional_init_for_megablocks_moe()

    def has_v1(self):
        if self.mlp_prefix:
            return "v1" in self.mlp
        else:
            return hasattr(self, "v1") and self.v1 is not None

    def _get_w1_for_mlp_prefix(self):
        return self.mlp["w1"]

    def _get_w2_for_mlp_prefix(self):
        return self.mlp["w2"]

    def _get_v1_for_mlp_prefix(self):
        return self.mlp["v1"]

    def _additional_init_for_megablocks_moe(self):
        # megablocks gather scatter
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.local_sort_end_bit = max(int(np.ceil(np.log2(self.num_local_experts))), 1)

    def _additional_init_for_megatron_moe(
        self,
    ):
        expert_rank = parallel_rank("expert") if parallel_rank("expert") else 0
        local_expert_indices_offset = expert_rank * self.num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]

        if self.token_dispatcher_type is MOETokenDispatcherType.AllToAll:
            token_dispatcher_cls = MoEAllToAllTokenDispatcher
        elif self.token_dispatcher_type is MOETokenDispatcherType.AllGather:
            token_dispatcher_cls = MoEAllGatherTokenDispatcher
        elif self.token_dispatcher_type is MOETokenDispatcherType.MindSpeedAllGather:
            token_dispatcher_cls = MindSpeedAllGatherTokenDispatcher
        elif self.token_dispatcher_type is MOETokenDispatcherType.MindSpeedAllToAll:
            token_dispatcher_cls = MindSpeedAllToAllTokenDispatcher

        self.token_dispatcher = token_dispatcher_cls(
            self.num_local_experts,
            local_expert_indices,
            self.num_experts,
            topk=self.topk,
            add_bias=self.use_bias,
            use_expert_parallelism=self.use_expert_parallelism,
        )

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

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        top_experts: torch.Tensor,
        se_fn1: Optional[Callable] = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_additional_input: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ):
        if self.implementer_type is MOEImplmenterType.Megatron:
            return self._forward_megatron(
                hidden_states, expert_weights, top_experts, se_fn1, se_fn2, se_fn2_additional_input
            )
        elif self.implementer_type is MOEImplmenterType.MegaBlocks:
            return self._forward_megablocks(
                hidden_states, expert_weights, top_experts, se_fn1, se_fn2, se_fn2_additional_input
            )

    def _forward_megatron(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,  # scores
        top_experts: torch.Tensor,  # indices
        se_fn1: Optional[Callable] = None,
        se_fn2: Optional[Callable] = None,
        se_fn2_additional_input: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ):
        expert_weights = expert_weights.view(-1, expert_weights.shape[-1])
        top_experts = top_experts.view(-1, top_experts.shape[-1])
        self.hidden_shape = hidden_states.shape
        dispatched_input, tokens_per_expert, se_intermediate = self.token_dispatcher.token_permutation(
            hidden_states, expert_weights, top_experts, se_fn1
        )

        if se_fn1 is not None and (EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP or se_intermediate is None):
            se_intermediate = se_fn1(hidden_states.view(-1, self.hidden_shape[-1]))

        expert_output = self.compute_expert(dispatched_input, tokens_per_expert)

        compute_fn_input = None
        if se_fn2_additional_input is not None:
            if isinstance(se_fn2_additional_input, (tuple, list)):
                compute_fn_input = (se_intermediate,) + tuple(se_fn2_additional_input)
            else:
                compute_fn_input = (se_intermediate, se_fn2_additional_input)
        elif se_fn2 is not None:
            compute_fn_input = (se_intermediate,)

        output, _, se_output = self.token_dispatcher.token_unpermutation(
            expert_output, se_fn2=se_fn2, se_fn2_input=compute_fn_input
        )
        if se_fn2 is not None and (EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP or se_output is None):
            se_output = se_fn2(*compute_fn_input)

        if se_output is not None:
            output += se_output.view(self.hidden_shape)
        return output

    def _forward_megablocks(
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
        # Permutation
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
            if EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                A2Aed_hidden_states, a2a_hs_handle, _ = all_to_all_with_compute_overlap(
                    permuted_hidden_states,
                    recv_counts,
                    send_counts,
                    self.expert_parallel_group,
                    True,
                    None,
                    None,
                )
                if se_fn1 is not None:
                    se_intermediate = se_fn1(hidden_states)
            else:
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

        # Compute expert
        _bin_ids = _bin_ids if self.num_local_experts > 1 else None
        expert_output = self.compute_expert(_hidden_states, _tokens_per_expert, _bin_ids)

        # Unpermutation
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
            if EnvSetting().MOE_DISABLE_SHARED_EXPERT_OVERLAP:
                expert_output, a2a_eo_handle, _ = all_to_all_with_compute_overlap(
                    expert_output,
                    send_counts,
                    recv_counts,
                    self.expert_parallel_group,
                    True,
                    None,
                    None,
                )
                if se_fn2 is not None:
                    se_output = se_fn2(*compute_fn_input)
                a2a_eo_handle.wait()
            else:
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

    def compute_expert(self, _hidden_states, _tokens_per_expert, bin_ids=None):
        w1, w2 = self.scale_grad(self.w1), self.scale_grad(self.w2)
        v1 = self.scale_grad(self.v1) if self.has_v1() else None

        # Compute expert
        if _hidden_states.nelement() == 0:
            # no valid token per expert & permuted local hidden states
            if not is_torch_npu_available():
                assert _tokens_per_expert.sum(dim=0) == 0
            expert_output = self.compute_expert_no_token(_hidden_states, w1, w2, v1)
        else:
            if self.num_local_experts > 1:
                expert_output = self.compute_moe(_hidden_states, _tokens_per_expert, w1, w2, v1, bin_ids)
            else:
                expert_output = self.compute_single_expert(_hidden_states, w1, w2, v1)

        return expert_output

    def compute_moe(self, hidden_states, tokens_per_expert, w1, w2, v1, bin_ids=None):
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
        if EnvSetting().DEBUG:
            print("rank:", rank(), "tokens_per_expert:", tokens_per_expert)
        if self.use_swiglu and not self.merge_w1_v1:
            x1 = gmm(hidden_states, w1, tokens_per_expert, trans_b=self.transpose_w1)
            x2 = gmm(hidden_states, v1, tokens_per_expert, trans_b=self.transpose_w1)
            intermediate_states = F.silu(x1) * x2
        else:
            fc1_output = gmm(hidden_states, w1, tokens_per_expert, trans_b=self.transpose_w1)
            if self.use_bias:
                assert bin_ids is not None
                fc1_output = bias_gather_add(fc1_output, self.b1, bin_ids)
            intermediate_states = self.activation(fc1_output)
        fc2_output = gmm(intermediate_states, w2, tokens_per_expert, trans_b=False)
        if self.use_bias:
            assert bin_ids is not None
            fc2_output = bias_gather_add(fc2_output, self.b2, bin_ids)
        expert_output = self.dropout(fc2_output)
        return expert_output

    def compute_single_expert(self, hidden_states, w1, w2, v1):
        if not self.transpose_w1:
            _w1 = w1.squeeze(0).transpose(0, 1)
        else:
            _w1 = w1.squeeze(0)

        if self.use_swiglu and not self.merge_w1_v1:
            if not self.transpose_w1:
                _v1 = v1.squeeze(0).transpose(0, 1)
            else:
                _v1 = v1.squeeze(0)
            x1 = F.linear(hidden_states, _w1, None)
            x2 = F.linear(hidden_states, _v1, None)
            intermediate_states = F.silu(x1) * x2
        else:
            fc1_output = F.linear(hidden_states, _w1, self.b1.squeeze(0) if self.use_bias else None)
            intermediate_states = self.activation(fc1_output)

        fc2_output = F.linear(
            intermediate_states, w2.squeeze(0).transpose(0, 1), self.b2.squeeze(0) if self.use_bias else None
        )
        expert_output = self.dropout(fc2_output)
        return expert_output

    def compute_expert_no_token(self, hidden_states, w1, w2, v1):
        w1 = w1.view(self.hidden_size, -1)
        w2 = w2.view(-1, self.hidden_size)
        if self.use_swiglu and not self.merge_w1_v1:
            v1 = v1.view(self.hidden_size, -1)
            x1 = torch.matmul(hidden_states, w1)
            x2 = torch.matmul(hidden_states, v1)
            intermediate_states = F.silu(x1) * x2
        else:
            fc1_output = torch.matmul(hidden_states, w1)
            intermediate_states = self.activation(fc1_output)
        if self.use_bias:
            raise NotImplementedError("Not support bias when there is no local token.")
        fc2_output = torch.matmul(intermediate_states, w2)
        return fc2_output


if EnvSetting().MOE_MLP_PREFIX:
    setattr(Grouped_GEMM_MoE, "w1", property(Grouped_GEMM_MoE._get_w1_for_mlp_prefix))
    setattr(Grouped_GEMM_MoE, "w2", property(Grouped_GEMM_MoE._get_w2_for_mlp_prefix))
    setattr(Grouped_GEMM_MoE, "v1", property(Grouped_GEMM_MoE._get_v1_for_mlp_prefix))
