# MODIFIED from torch.distributed.fsdp._runtime_utils
import logging
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple, no_type_check

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    TrainingState,
    _assert_in_training_states,
    _FSDPState,
    _is_composable,
    _log_post_backward_hook,
    _no_dispatch_record_stream,
    _set_fsdp_flattened,
)
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp._runtime_utils import (
    HOMOGENEOUS_ATTR_NAMES,
    _accumulate_sharded_grad,
    _cast_buffers_to_dtype_and_device,
    _div_if_needed,
    _get_buffers_and_dtypes_for_computation,
    _get_reduce_scatter_tensors,
    _init_device_mesh,
    _lazy_init,
    _low_precision_hook_enabled,
    _post_backward_reshard,
    _post_forward,
    _post_forward_reshard,
    _post_reduce_grad_callback,
    _pre_forward_unshard,
    _reduce_grad_no_shard,
    _register_post_backward_hook,
    _register_post_backward_reshard_only_hook,
    _reset_flat_param_grad_info_if_needed,
    _root_cast_forward_input,
    _validate_and_get_hybrid_shard_state,
    _wait_for_computation_stream,
)
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy, HandleTrainingState
from torch.distributed.utils import _cast_forward_inputs, _p_assert, _to_kwargs

logger = logging.getLogger(__name__)

SYNC_MODE_CONTROL = False


@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,
    root_module: nn.Module,
) -> None:
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    _validate_and_get_hybrid_shard_state(root_module)
    attr_name_to_values: Dict[str, Set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    root_state._device_mesh = _init_device_mesh(root_state)
    # Update _has_optim_in_backward for each handle.
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError("FSDP optimizer in backward only supported with use_orig_params=True!")
        handle._has_optim_in_backward = flat_param._params is not None and any(
            hasattr(param, "_in_backward_optimizers") for param in flat_param._params
        )
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        # Relax the assert for non-root FSDP instances in case the nested
        # initialized module is wrapped again in FSDP later (e.g. after
        # training to run inference)
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been " "set yet or should have been set to `False`",
        )
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        # HACK Share the average stream across FSDP instances.
        fsdp_state._average_stream = root_state._average_stream
        if fsdp_state.use_local_sgd and fsdp_state.local_sgd_cpu_offload:
            fsdp_state._D2H_stream = root_state._D2H_stream
            fsdp_state._H2D_stream = root_state._H2D_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        fsdp_state._device_mesh = root_state._device_mesh
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(f"Expects one homogeneous value for {attr_name} but got {attr_values}")


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES for fsdp_state in state._all_fsdp_states
    )
    # HACK Determine whether to use local sgd
    uses_local_sgd = any(fsdp_state.use_local_sgd for fsdp_state in state._all_fsdp_states)
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # HACK Add and modify some streams.
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = (
        state._device_handle.Stream() if (uses_hybrid_sharding and not uses_local_sgd) else state._default_stream
    )
    # Stream to run HSDP's average parameters as async (if using HSDP)
    state._average_stream = (
        state._device_handle.Stream() if (uses_hybrid_sharding and uses_local_sgd) else state._default_stream
    )
    state._D2H_stream = (
        state._device_handle.Stream()
        if state.use_local_sgd and (state.gta_reducer is not None or state.use_outer_optim)
        else state._default_stream
    )
    state._H2D_stream = (
        state._device_handle.Stream()
        if state.use_local_sgd and (state.gta_reducer is not None or state.use_outer_optim)
        else state._default_stream
    )


# HACK Lazy init outer optimizer.
@no_type_check
def _lazy_init_outer_optimizer(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    cpu_init: bool = False,
) -> None:
    if cpu_init:
        cpu_flat_param = handle.flat_param.cpu().detach().clone()
        pinned_cpu_flat_param = cpu_flat_param.pin_memory()  # for non-blocking copy
        state.last_synced_params = torch.nn.Parameter(pinned_cpu_flat_param)
    else:
        state.last_synced_params = torch.nn.Parameter(handle.flat_param.detach().clone())
    _set_fsdp_flattened(state.last_synced_params)
    if state.outer_optim_class is not None:
        state.outer_optimizer = state.outer_optim_class([state.last_synced_params], **state.outer_optim_kwargs)
    else:
        state.outer_optimizer = None


# HACK Determine whether synchronization is required.
def _sync_if_needed(
    state: _FSDPState,
    module: nn.Module,
) -> bool:
    if not module.training:
        return False
    if not state.use_local_sgd:
        return False
    if not state._handle.flat_param.requires_grad:
        return False
    if state.temp_step % state.gradient_accumulation_steps:  # only average once for the same `global_step`
        return False
    if state.global_step < state.local_sgd_warmup_steps:
        if state.use_async:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            state.last_total_global_step = total_global_step.item()
            state.last_sync_timestamp = time.time()
        return False
    if state.use_async:
        if state.last_sync_timestamp is None:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            state.last_total_global_step = total_global_step.item()
            state.last_sync_timestamp = time.time()
            return True
        elif time.time() - state.last_sync_timestamp < state.local_sgd_sync_time:
            return False
        else:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            diff_global_steps = total_global_step.item() - state.last_total_global_step
            if diff_global_steps < state.min_total_global_steps:
                # Should we skip this synchronization or keep looping until the synchronization is completed?
                state.last_sync_timestamp = time.time()
                return False
            else:
                state.last_sync_timestamp = time.time()
                state.last_total_global_step = total_global_step.item()
                return True
    else:
        if (state.global_step - state.local_sgd_warmup_steps) % state.local_sgd_sync_interval:
            return False
        return True


# HACK sync sharded parameters among nodes.
# TODO make skip_anomaly fully compatible with non reducer reduction
@no_type_check
def _sync_sharded_params(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    with state._device_handle.stream(state._average_stream):
        pseudo_gradient = None
        if state.use_async and state.use_step_weight:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            step_weight = state.global_step / total_global_step.item()

        if state.gta_reducer is None and not state.use_outer_optim:
            with torch.no_grad():
                if state.use_async and state.use_step_weight:
                    handle.flat_param.data *= step_weight
                else:
                    handle.flat_param.data /= dist.get_world_size(state._inter_node_pg)
                dist.all_reduce(handle.flat_param.data, group=state._inter_node_pg)
        else:
            if state.last_synced_params is None:
                _lazy_init_outer_optimizer(state, handle)

    if state.local_sgd_cpu_offload:
        param_device = handle.flat_param.device
        if state.gta_reducer is not None or state.use_outer_optim:
            with state._device_handle.stream(state._H2D_stream):
                with torch.no_grad():
                    state.last_synced_params.data = state.last_synced_params.data.to(param_device, non_blocking=True)

            # wait for last synced params to be moved to cuda
            state._average_stream.wait_stream(state._H2D_stream)

        # then we do async optim H2D
        if state.use_outer_optim:
            with state._device_handle.stream(state._H2D_stream):
                # TODO separate opt and last sync params movement
                for opt_state in state.outer_optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k].data = v.data.to(param_device, non_blocking=True)

    with state._device_handle.stream(state._average_stream):
        if state.gta_reducer is not None or state.use_outer_optim:
            with torch.no_grad():
                pseudo_gradient = state.last_synced_params.data - handle.flat_param.data
                if state.local_sgd_skip_anomaly or state.pseudo_gradnorm_reduce:
                    pseudo_gradnorm = torch.linalg.vector_norm(pseudo_gradient, 2).pow_(2)
                    dist.all_reduce(pseudo_gradnorm, group=state.process_group, op=torch.distributed.ReduceOp.SUM)
                    pseudo_gradnorm.pow_(0.5)

                if state.local_sgd_skip_anomaly:
                    # FIXME this is stuck by D2H
                    gradnorm_value = pseudo_gradnorm
                    if state.local_sgd_anomaly_detector.is_outlier(gradnorm_value):
                        # nullify this work's update
                        pseudo_gradient = torch.zeros_like(
                            state.last_synced_params.data, dtype=pseudo_gradient.dtype, device=pseudo_gradient.device
                        )
                        pseudo_gradnorm += 1e6
                        if dist.get_rank(state.process_group) == 0:
                            print(
                                f"Full Shard group [{dist.get_rank(state._inter_node_pg)}] step [{state.global_step}] ",
                                f"Pseudo Gradnorm: {gradnorm_value} deemed outlier",
                            )
                    state.local_sgd_anomaly_detector.update(gradnorm_value)

                if state.gta_reducer is None:
                    if state.use_async and state.use_step_weight:
                        pseudo_gradient *= step_weight
                    else:
                        pseudo_gradient /= dist.get_world_size(state._inter_node_pg)
                    dist.all_reduce(pseudo_gradient, group=state._inter_node_pg)
                else:
                    reducer_kwargs = {}
                    if state.pseudo_gradnorm_reduce:
                        # penalty on local copies with large grad norm
                        reducer_kwargs["weight"] = -pseudo_gradnorm
                    if state.use_async and state.use_step_weight:
                        # penalty on local copies with step weight
                        reducer_kwargs["step_weight"] = step_weight
                        reducer_kwargs["step_weight_ratio"] = state.step_weight_ratio
                    state.gta_reducer.reduce_tensor(pseudo_gradient, **reducer_kwargs)
        if pseudo_gradient is not None:
            if state.clip_pseudo_grad is not None:
                total_norm = torch.linalg.vector_norm(pseudo_gradient, 2).pow_(2)
                dist.all_reduce(total_norm, group=state.process_group)
                total_norm.pow_(0.5)
                clip_coef = state.clip_pseudo_grad / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                pseudo_gradient.detach().mul_(clip_coef_clamped)
                if dist.get_rank(state.process_group) == 0:
                    logger.debug(
                        f"rank[{dist.get_rank()}]: "
                        f"total_norm:{total_norm}, "
                        f"clip_pseudo_grad:{state.clip_pseudo_grad}, "
                        f"clip_coef_clamped:{clip_coef_clamped}."
                    )
        # TODO check if optimizer states are casted by FSDP
        if pseudo_gradient is not None:
            if not (state.local_sgd_skip_anomaly and pseudo_gradient.abs().sum() < 1e-6):
                if state.use_outer_optim:
                    if state.local_sgd_cpu_offload:
                        state._average_stream.wait_stream(state._H2D_stream)
                    state.last_synced_params.grad = pseudo_gradient.to(state.last_synced_params.device)
                    state.outer_optimizer.step()
                    state.outer_optimizer.zero_grad()
                else:
                    state.last_synced_params.data = state.last_synced_params.data - pseudo_gradient.to(
                        state.last_synced_params.device
                    )
            # The _average_stream ensures that non-blocking copy is correct.
            handle.flat_param.data.copy_(state.last_synced_params.data, non_blocking=True)

    if state.local_sgd_cpu_offload:
        state._D2H_stream.wait_stream(state._average_stream)
        with state._device_handle.stream(state._D2H_stream):
            if state.use_outer_optim:
                for opt_state in state.outer_optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k].data = v.data.to("cpu", non_blocking=True)
            if state.gta_reducer is not None or state.use_outer_optim:
                with torch.no_grad():
                    state.last_synced_params.data = state.last_synced_params.data.to("cpu", non_blocking=True)


@no_type_check
def _pre_forward(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
        # For `fully_shard` + `checkpoint`, skip pre-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # For both checkpoint implementations, we do not need to re-cast
            # inputs here since they will be checkpointed in the low precision
            # either by AC or normally by autograd as long as the AC region is
            # nested within FSDP
            return args, kwargs
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD

        # HACK The main code for averaging parameters among nodes.
        if SYNC_MODE_CONTROL:
            _sync_sharded_params(state, handle)

            _wait_for_computation_stream(
                state._average_stream,
                state._unshard_stream,
                state._pre_unshard_stream,
            )
        if unshard_fn is not None:
            unshard_fn(state, handle)
        # Register post-backward hooks to reshard the parameters and reduce-scatter
        # their gradients. They must be re-registered every forward pass in case
        # the `grad_fn` is mutated.
        _register_post_backward_hook(state, handle)
        # We have to reallocate the _cpu_grad if optimizer overlap
        # set the grad to None in the backward pass.
        if handle and handle._offload_params and handle.flat_param._cpu_grad is None:
            handle.flat_param._cpu_grad = torch.zeros_like(
                handle.flat_param._local_shard, device=torch.device("cpu")
            ).pin_memory()

        should_cast_forward_inputs = state._handle and not state._handle._force_full_precision

        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return args, kwargs


@no_type_check
@torch.no_grad()
def _post_backward_hook(state: _FSDPState, handle: FlatParamHandle, flat_param, *unused: Any):  # type: ignore
    _log_post_backward_hook(state, handle, logger)  # type: ignore
    flat_param, flat_param._post_backward_called = handle.flat_param, True  # type: ignore
    with torch.autograd.profiler.record_function("FullyShardedDataParallel._post_backward_hook"):  # type: ignore
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])  # type: ignore
        _p_assert(  # type: ignore
            handle._training_state in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",  # type: ignore
        )  # type: ignore
        handle._training_state = HandleTrainingState.BACKWARD_POST  # type: ignore
        if flat_param.grad is None:  # type: ignore
            return  # type: ignore
        if flat_param.grad.requires_grad:  # type: ignore
            raise RuntimeError("FSDP does not support gradients of gradients")  # type: ignore
        _post_backward_reshard(state, handle)  # type: ignore
        if not state._sync_gradients:  # type: ignore
            if handle._use_orig_params:  # type: ignore
                handle._use_unsharded_grad_views()  # type: ignore
            return  # type: ignore
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():  # type: ignore
            state._post_backward_stream.wait_stream(state._device_handle.current_stream())  # type: ignore
        with state._device_handle.stream(state._post_backward_stream):  # type: ignore
            autograd_computed_grad = flat_param.grad.data  # type: ignore
            if (  # type: ignore
                not _low_precision_hook_enabled(state)  # type: ignore
                and flat_param.grad.dtype != handle._reduce_dtype  # type: ignore
                and not handle._force_full_precision  # type: ignore
            ):  # type: ignore
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)  # type: ignore
            # HACK check gradnorm outlier
            if state.skip_anomaly:  # type: ignore
                flat_param_gradnorm = torch.linalg.vector_norm(flat_param.grad.data, 2)
                if state.anomaly_detector.is_outlier(flat_param_gradnorm):  # type: ignore
                    print(f"Rank: [{dist.get_rank()}] Gradnorm: {flat_param_gradnorm} deemed outlier")  # type: ignore
                    flat_param.grad.data = torch.zeros_like(  # type: ignore
                        flat_param.grad.data, dtype=handle._reduce_dtype, device=flat_param.grad.device  # type: ignore
                    )  # type: ignore
                state.anomaly_detector.update(
                    flat_param_gradnorm
                )  # TODO verify whether not to push anomaly is good # type: ignore
            _reduce_grad(state, handle) if handle.uses_sharded_strategy else _reduce_grad_no_shard(state, handle)
            _no_dispatch_record_stream(autograd_computed_grad, state._post_backward_stream)  # type: ignore


@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param
    # HACK Fobidden grads all-reduce among nodes if `use_local_sgd` and accumulate `global_step`
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    if state.use_local_sgd:
        uses_hybrid_sharded_strategy = (
            uses_hybrid_sharded_strategy and not state.global_step >= state.local_sgd_warmup_steps
        )
        state.temp_step += 1
        if not state.temp_step % state.gradient_accumulation_steps:
            state.global_step += 1
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=state.process_group,
        )
        if uses_hybrid_sharded_strategy:
            state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                # Since the new sharded gradient is produced in the post-
                # backward stream and consumed in the all-reduce stream,
                # inform the caching allocator
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
        # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._root_pre_forward"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, "Expects a root FSDP to have been set")
        if not state._is_root:
            # Always cast forward inputs in the root of this local FSDP unit for mixed
            # precision, as this is where mixed precision could be configed.
            # This is more useful for auto wrapping that is recommended in composable path.
            # For manual wrapping, cast forward inputs on each local FSDP unit root will
            # increase some overhead, so not turned on for model wrapper path right now where
            # manual wrapping is more broadly used.
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return args, kwargs

        # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
        # are in full precision and if we should cast them back to lower precision, which happens when
        # exiting eval() mode.
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            should_cast_buffers_to_full_prec = True

        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(
                buffers=dict(module.named_buffers()).values(),
                buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                device=state.compute_device,
            )
            # This flag is only set when we cast buffers to full precision, to avoid the
            # CPU overhead that can stem from retrieving all buffers and their types in the
            # following else branch.
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, "_needs_buffer_dtype_restore_check", False):
            # Check if buffers are in full precision and we need to cast them
            # back down.
            (
                buffers,
                buffer_dtypes_for_computation,
            ) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any(
                    buffer.dtype != buffer_dtype_for_computation
                    for buffer, buffer_dtype_for_computation in zip(buffers, buffer_dtypes_for_computation)
                ):
                    # Assume we have to cast everything if there is one mismatch
                    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes_for_computation, state.compute_device)
            # We don't have to check this again until we cast buffers to full precision again.
            state._needs_buffer_dtype_restore_check = False

        if state.forward_prefetch:
            handles = []
            for fsdp_state in state._all_fsdp_states:
                if fsdp_state._handle:
                    handles.append(fsdp_state._handle)
            for handle in handles:
                handle._needs_pre_forward_unshard = True
        # HACK Control the calculation streams corresponding to local sgd
        global SYNC_MODE_CONTROL
        SYNC_MODE_CONTROL = _sync_if_needed(state, module)
        if SYNC_MODE_CONTROL:
            state._average_stream.wait_stream(state._device_handle.current_stream())
        else:
            _wait_for_computation_stream(
                state._device_handle.current_stream(),
                state._unshard_stream,
                state._pre_unshard_stream,
            )
        _reset_flat_param_grad_info_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # TODO: Do not use the side stream for tensor copies for now; investigate
        # the perf with/without it.
        with torch.profiler.record_function("FullyShardedDataParallel._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, state.compute_device, False)
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]

        return _root_cast_forward_input(state, module, args, kwargs)


# HACK Use `_root_pre_forward` and `_pre_forward` modified here.
def forward(self, *args: Any, **kwargs: Any) -> Any:
    """
    Runs the forward pass for the wrapped module, inserting FSDP-specific
    pre- and post-forward sharding logic.
    """
    handle = self._handle
    with torch.autograd.profiler.record_function("FullyShardedDataParallel.forward"):
        args, kwargs = _root_pre_forward(self, self, args, kwargs)
        unused = None
        args, kwargs = _pre_forward(
            self,
            handle,
            _pre_forward_unshard,
            self._fsdp_wrapped_module,
            args,
            kwargs,
        )
        if handle:
            _p_assert(
                handle.flat_param.device == self.compute_device,
                "Expected `FlatParameter` to be on the compute device "
                f"{self.compute_device} but got {handle.flat_param.device}",
            )
        output = self._fsdp_wrapped_module(*args, **kwargs)
        return _post_forward(self, handle, _post_forward_reshard, self, unused, output)
