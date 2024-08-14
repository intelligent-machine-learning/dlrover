import functools
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp._utils import _contains_batchnorm, _override_batchnorm_mixed_precision
from torch.distributed.fsdp.flatten_params_wrapper import FlatParameter, FlattenParamsWrapper
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    _PARAM_BROADCAST_BUCKET_SIZE,
    FSDP_WRAPPED_MODULE,
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
    TrainingState_,
    _default_meta_device_init_fn,
    _ExecOrderData,
    _get_default_cuda_device,
)
from torch.distributed.fsdp.wrap import _or_policy, _recursive_wrap, _wrap_batchnorm_individually
from torch.distributed.utils import _sync_params_and_buffers
from torch.nn.parameter import Parameter

from atorch.distributed.distributed import parallel_group, parallel_group_and_ranks

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake
except ImportError:
    _TORCHDISTX_AVAIL = False


class FSDPWithDDP(FullyShardedDataParallel):
    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Callable] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super(FullyShardedDataParallel, self).__init__()
        # Validate the ignored modules and derive the ignored parameters/buffers
        ignored_modules = self._get_ignored_modules(module, ignored_modules)
        ignored_params, ignored_param_names = self._get_ignored_params(module, ignored_modules)
        buffer_names = self._get_buffer_names(module)
        # Compute the names to ignore for full state dict cloning (i.e. those
        # of the ignored modules' parameters and of all modules' buffers)
        self._ignored_param_names = ignored_param_names
        self._buffer_names = buffer_names
        # NOTE: Since the names are computed at construction time, if the user
        # changes them later, then FSDP will not properly ignore them. However,
        # the `FlatParameter` implementation already relies on this assumption.
        # We do this at construction time since we want the fully prefixed
        # parameter names matching the keys in the model state dict (namely,
        # including the wrapped module's name in the prefix), which may be done
        # most non-intrusively here before flattening.

        # if auto_wrap_policy is specified, submodules should not be
        # already wrapped, otherwise we'd attempt to double wrap them resulting
        # in errors.
        if auto_wrap_policy is not None:
            self._check_wrapped(
                module,
                check_fn=lambda mod: not isinstance(mod, FullyShardedDataParallel),
                err_fn=lambda mod: f"Expected {mod} to NOT be FullyShardedDataParallel if auto_wrap is enabled.",
            )
            if mixed_precision is not None and _contains_batchnorm(module):
                _override_batchnorm_mixed_precision(module)
                policy_to_use = functools.partial(_or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy])
                warnings.warn(
                    "Mixed precision was specified for FSDP module with"
                    " batchnorm submodules wrapped via ``auto_wrap_policy``."
                    " BatchNorm units will be wrapped as a separate FSDP unit,"
                    " with mixed_precision disabled (i.e. set to ``None``)"
                    " as several BatchNorm kernels would raise errors when"
                    " operating on reduced precision inputs."
                )
            else:
                policy_to_use = auto_wrap_policy  # type: ignore
            _recursive_wrap(
                module,
                auto_wrap_policy=policy_to_use,
                wrapper_cls=FullyShardedDataParallel,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                # Note that we have the recursive_wrap skip wrapping for
                # the outermost (this) module otherwise it will result in a
                # double-wrap causing issues.
                only_wrap_children=True,
                # FSDP arguments follow.
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
                device_id=device_id,
                sync_module_states=sync_module_states,
            )

        self.process_group = process_group or _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        if device_id is not None:
            self.device_id = device_id if isinstance(device_id, torch.device) else torch.device(device_id)
            # If user passed in something like torch.device("cuda"),
            # device index of current device is unclear, make it explicit.
            if self.device_id == torch.device("cuda"):
                warnings.warn(
                    f"Passed in {self.device_id} does not have explicit index, "
                    f"setting it to current index: {torch.cuda.current_device()}. "
                    "If this is not correct, please explicitly call torch.cuda.set_device()"
                    "before FSDP initialization or pass in explicit device index as device_id argument."
                )
                self.device_id = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device_id = None

        is_meta_module = any(p.is_meta for p in module.parameters())
        is_torchdistX_deferred_init = (
            not is_meta_module and _TORCHDISTX_AVAIL and any(fake.is_fake(p) for p in module.parameters())
        )

        def _run_param_init_fn(module):
            # Call user-specified initialization function.
            if not callable(param_init_fn):
                raise ValueError(f"Expected {param_init_fn} to be callable, but got {type(param_init_fn)}")
            param_init_fn(module)

        if is_meta_module:
            if param_init_fn is not None:
                _run_param_init_fn(module)
            else:
                # Call default initialization function that is dependent on
                # reset_parameters.
                _default_meta_device_init_fn(module)
        elif is_torchdistX_deferred_init:
            assert _TORCHDISTX_AVAIL, "Got torchdistX initialized module but torchdistX lib is not available."
            if param_init_fn is not None:
                _run_param_init_fn(module)
            else:
                # Call default torchdistX initialization function. Omit re-initialization of FSDP submodules
                # which is unnecessary.
                check_fn = lambda k: not isinstance(k, FullyShardedDataParallel)  # noqa: E731
                deferred_init.materialize_module(module, check_fn=check_fn)

        # Check that module was placed onto a single device.
        module_devices = set(
            p.device for p in module.parameters() if p not in ignored_params and not isinstance(p, FlatParameter)
        )

        if len(module_devices) > 1:
            raise RuntimeError(f"FSDP only supports single device modules, but got params on {module_devices}")

        # Move module appropriately depending on device_id and whether module is on CPU.
        self._move_module_if_needed(module)

        # device for computation, if module is on GPU, use module.device;
        # if module is on CPU, use current device;
        self.compute_device = _get_default_cuda_device(module)

        # if device_id is specified, ensure it is the same
        assert (
            self.device_id is None or self.compute_device == self.device_id
        ), f"Inconsistent compute_device and device_id: {self.compute_device} vs {self.device_id}"

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState_.IDLE

        # setting two factors to avoid underflow and overflow
        self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

        self.numel_padded_per_param: List[int] = []
        self.cpu_offload = cpu_offload or CPUOffload()
        self.backward_prefetch = backward_prefetch
        self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
        self.mixed_precision = mixed_precision
        # Original buffer type (mapping since all buffers may not be of same type). In
        # the case of mixed precision training, this is used to restore buffers
        # to their original type (which may not be the same as that of the
        # parameters in the model) when checkpointing.
        self._orig_buffer_dtypes: Dict[str, torch.dtype] = {}

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = [p for p in module.parameters() if p not in ignored_params and not isinstance(p, FlatParameter)]

        if params != [] and params[0].device == torch.device("cpu"):
            raise ValueError(
                "Module has CPU parameters, but sync_module_states=True is specified."
                "This only works for GPU module, please specify `device_id` argument or move"
                " module to GPU before init."
            )
        # Collect buffers we have to synchronize, avoiding buffers that have already
        # been synchronized to avoid redundant synchronization.
        bufs_to_sync = []
        for buf in module.buffers():
            if not getattr(buf, "_fsdp_has_been_sync", False):
                buf._fsdp_has_been_sync = True
                bufs_to_sync.append(buf.detach())

        states_to_sync = [param.detach() for param in params]
        states_to_sync.extend(bufs_to_sync)

        if sync_module_states:
            _sync_params_and_buffers(
                process_group=self.process_group,
                module_states=states_to_sync,
                # Same bucket size as DDP
                broadcast_bucket_size=_PARAM_BROADCAST_BUCKET_SIZE,
                src=0,
            )

        _sync_params_and_buffers(
            # use DDP_GROUP
            process_group=parallel_group("data"),
            module_states=states_to_sync,
            # Same bucket size as DDP
            broadcast_bucket_size=_PARAM_BROADCAST_BUCKET_SIZE,
            src=0,
        )

        self._fsdp_wrapped_module: FlattenParamsWrapper = FlattenParamsWrapper(module, param_list=params)
        assert getattr(self, FSDP_WRAPPED_MODULE) is self._fsdp_wrapped_module
        del module  # free original module in case it helps garbage collection
        if self._fsdp_wrapped_module.flat_param is not None:
            self.params = [self._fsdp_wrapped_module.flat_param]
        else:
            self.params = []

        # Shard module parameters in place
        self._shard_parameters()

        # Make sure all parameters are sharded.
        for n, p in self.named_parameters():
            if p not in ignored_params and not isinstance(p, FlatParameter):
                raise RuntimeError(f"found unflattened parameter: {n} ; {p.size()} {p.__class__}")
        self._reset_lazy_init()

        # Flag indicating if we require gradient reduction in the backward
        # pass (set to `False` in the `no_sync()` context manager)
        self._require_backward_grad_sync: bool = True

        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config = FullStateDictConfig()

        # FSDP currently provides three different state_dicts. The actual
        # state_dict that will be saved/loaded is decided by
        # self._state_dict_type. And the main logic of each state_dict is
        # implemented in the hook. Therefore, for each hook (post-save and
        # pre-load), there is a dispatcher dictionary to dispatch the execution
        # flow to the correct implementation.
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_post_state_dict_hook,
        }
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook, with_module=True)
        self._pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_pre_load_state_dict_hook,
        }
        self.register_load_state_dict_post_hook(self._post_load_state_dict_hook)
        self._post_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_post_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_post_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_post_load_state_dict_hook,
        }

        # Flag to guard against preparing gradients multiple times per backward pass.
        self._pre_backward_hook_has_run = False
        # Used for prefetching all gather full params in post backward hook
        self._need_rebuild_full_params = False

        # If specified, offload parameter shard to CPU.
        if self.cpu_offload.offload_params:
            for p in self.params:
                self._offload_to_cpu(p)

        # For validating execution order across ranks
        self._exec_order_data = _ExecOrderData()

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
        """
        Hook _post_backward_hook to support FSDP + DDP
        """

        def p_assert(cond: Any, s: Any) -> None:
            """This is used as an alternate to ``assert`` when in the backward context
            to print the error message ``s`` since otherwise, it is swallowed."""
            if not cond:
                print(s)
                raise AssertionError

        # First hook callback will see PRE state. If we have multiple params,
        # then subsequent hook callbacks will see POST state.
        self._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST])
        self.training_state = TrainingState_.BACKWARD_POST
        if param.grad is None:
            return

        if param.grad.requires_grad:
            raise RuntimeError("FSDP only works with gradients that don't require gradients")

        if self._require_backward_grad_sync or self.sharding_strategy == ShardingStrategy.FULL_SHARD:
            # We free full parameters unless we are in `no_sync()` (i.e. when
            # `_require_backward_grad_sync=False`) and not using the
            # `FULL_SHARD` strategy. If we are not using the `FULL_SHARD`
            # strategy (e.g. instead using `SHARD_GRAD_OP`), then we keep the
            # full parameters in memory and save network overhead.
            self._free_full_params(cast(List[FlatParameter], [param]))

        if self._mixed_precision_enabled_for_params():
            # Noop if reshard_after_forward=True because we'd free the param
            # shard when rebuilding the full params in the pre_beckward_hook.
            self._free_mp_shard(cast(List[FlatParameter], [param]))

        # Switch to local shard after backward. Note that
        # when CPU offload is enabled, _use_param_local_shard implicitly
        # offloads the local shard to CPU by making p.data point to
        # p._local_shard, which would reside on CPU.
        self._use_param_local_shard(cast(List[FlatParameter], [param]))

        # Prefetch previous layer's full params in backward pass post backward hook,
        # If next layer's backward computation is done and full params are freed,
        # no need to prefetch the full params again.
        # Only prefetch full params if any of the next layer's outputs requires grad
        if self._need_prefetch_post_backward_hook():
            self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]
            # Next layer's computation will start right after this all_gather,
            # Wait for all_gather to finish before computation.
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

        if not self._require_backward_grad_sync:
            return

        # Wait for all work in the current stream to finish, then start the
        # reductions in post_backward stream.
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._streams["post_backward"]):
            orig_grad_data = param.grad.data
            if self._mixed_precision_enabled_for_reduce():
                # Cast gradient to precision in which it should be communicated.
                # TODO: Make this a communication hook when communication hooks
                # are implemented for FSDP. Note that this is a noop if the
                # reduce_dtype matches the param dtype.
                param.grad.data = param.grad.data.to(self.mixed_precision.reduce_dtype)  # type: ignore

            if self.gradient_predivide_factor > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                param.grad.div_(self.gradient_predivide_factor)

            grad = param.grad.data
            if param._is_sharded:  # type: ignore[attr-defined]
                # We clear `param.grad` to permit repeated gradient
                # computations when this FSDP module is called multiple times.
                # This is to avoid a race among multiple re-entrant backward
                # passes. For example, the second backward pass computation
                # precedes ahead of the first backward pass reduction, which is
                # possible since the reduction is in a different stream and is
                # async. Then, the first backward pass may be incorrectly
                # reducing the second backward pass's `param.grad`.
                # The reduced gradients are accumulated in
                # `param._saved_grad_shard`, and the gradient reductions can
                # happen in arbitrary order, though we tolerate this due to the
                # (approximate) commutativity of floating-point addition.
                param.grad = None
                grad_flatten = torch.flatten(grad)
                chunks = list(grad_flatten.chunk(self.world_size))
                num_pad = self.world_size * chunks[0].numel() - grad.numel()
                input_flattened = F.pad(grad_flatten, [0, num_pad])
                output = torch.zeros_like(chunks[0])
                dist._reduce_scatter_base(output, input_flattened, group=self.process_group)
                dist.all_reduce(output, group=parallel_group("data"))
                _, ddp_group_ranks = parallel_group_and_ranks("data")
                if self.gradient_postdivide_factor > 1:
                    # Average grad by world_size for consistency with PyTorch DDP.
                    output.div_(self.gradient_postdivide_factor * len(ddp_group_ranks))
                else:
                    output /= len(ddp_group_ranks)

                # Note that we need to cast grads back to the full precision if
                # 1) parameters were in reduced precision during fwd, as grads
                # would thus be in this reduced precision, or
                # 2) parameters did not have precision reduced, but grads
                # had reduced precision for communication.
                if self._mixed_precision_enabled_for_params() or self._mixed_precision_enabled_for_reduce():
                    # Cast gradients back to the full parameter precision so that
                    # optimizer.step() happens in full precision.
                    orig_param_grad_data = output
                    output.data = output.data.to(dtype=param.data.dtype)
                    # Don't let this memory get reused until after the transfer.
                    orig_param_grad_data.record_stream(torch.cuda.current_stream())

                # To support gradient accumulation outside `no_sync()`, we save
                # the gradient data to `param._saved_grad_shard` before the
                # backward pass, accumulate gradients into it here, and set
                # `param.grad` with the accumulated value at the end of the
                # backward pass in preparation for the optimizer step.
                accumulate_grad = hasattr(param, "_saved_grad_shard")
                if accumulate_grad:
                    p_assert(
                        param._saved_grad_shard.shape == output.shape,  # type: ignore[attr-defined]
                        "Shape mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                        f"existing grad shape={param._saved_grad_shard.shape} "
                        f"new grad shape={output.shape}",  # type: ignore[attr-defined]
                    )
                    p_assert(
                        param._saved_grad_shard.device == output.device,  # type: ignore[attr-defined]
                        "Device mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                        f"existing grad device={param._saved_grad_shard.device} "
                        f"new grad device={output.device}",  # type: ignore[attr-defined]
                    )
                    param._saved_grad_shard += output  # type: ignore[attr-defined]
                else:
                    param._saved_grad_shard = output  # type: ignore[attr-defined]
                grad = param._saved_grad_shard  # type: ignore[attr-defined]
            else:
                # Currently the way for _is_sharded to be False is if
                # world_size == 1 or sharding_strategy is NO_SHARD.
                assert (
                    self.world_size == 1 or self.sharding_strategy == ShardingStrategy.NO_SHARD
                ), "Currently the way for _is_sharded to be False is \
                        world_size == 1 or sharding_stratagy is set to be NO_SHARD"
                if self.sharding_strategy == ShardingStrategy.NO_SHARD:
                    dist.all_reduce(param.grad, group=self.process_group)
                    if self.gradient_postdivide_factor > 1:
                        # Average grad by world_size for consistency with PyTorch DDP.
                        param.grad.div_(self.gradient_postdivide_factor)
                # Note that we need to cast grads back to the full precision if
                # 1) parameters were in reduced precision during fwd, as grads
                # would thus be in this reduced precision, or
                # 2) parameters did not have precision reduced, but grads
                # had reduced precision for communication.
                if self._mixed_precision_enabled_for_params() or self._mixed_precision_enabled_for_reduce():
                    # Cast gradients back to the full parameter precision so that
                    # optimizer.step() happens in full precision.
                    orig_param_grad_data = param.grad.data
                    param.grad.data = param.grad.data.to(dtype=param.data.dtype)
                    # Don't let this memory get reused until after the transfer.
                    orig_param_grad_data.record_stream(torch.cuda.current_stream())

            # Regardless of sharding or not, offload the grad to CPU if we are
            # offloading params. This is so param and grad reside on same device
            # which is needed for the optimizer step.
            if self.cpu_offload.offload_params:
                # We specify non_blocking=True
                # and ensure the appropriate synchronization is done by waiting
                # streams in _wait_for_post_backward.
                param._cpu_grad.copy_(grad.detach(), non_blocking=True)  # type: ignore[attr-defined]
                # Don't let this memory get reused until after the transfer.
                grad.data.record_stream(torch.cuda.current_stream())

            # After _post_backward_hook returns, orig_grad_data will eventually
            # go out of scope, at which point it could otherwise be freed for
            # further reuse by the main stream while the div/reduce_scatter/copy
            # are underway in the post_backward stream. See:
            # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
            orig_grad_data.record_stream(self._streams["post_backward"])
