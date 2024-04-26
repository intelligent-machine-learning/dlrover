# MODIFIED from torch.distributed.fsdp._init_utils and torch.distributed.fsdp.fully_sharded_data_parallel
from typing import Callable, Iterable, Optional, Type, Union, no_type_check

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, mesh_resources
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
    _check_orig_params_flattened,
    _check_single_device_module,
    _get_compute_device,
    _get_device_from_device_id,
    _get_orig_params,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_ignored_module_states,
    _init_param_handle_from_params,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    _materialize_meta_module,
    _materialize_with_param_init_fn,
    _move_module_to_device,
    _need_to_materialize_module,
    _sync_module_params_and_buffers,
)
from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
from torch.distributed.fsdp._unshard_param_utils import _register_flat_param
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim import Optimizer
from typing_extensions import Literal

# GTA related import
from atorch.local_sgd.reduce_methods import GTAReducer, LinearReducer

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False


def fsdp_inits(
    self,
    module: nn.Module,
    process_group: ProcessGroupType = None,
    sharding_strategy: Optional[ShardingStrategy] = None,
    cpu_offload: Optional[CPUOffload] = None,
    auto_wrap_policy: Optional[Union[Callable, ModuleWrapPolicy]] = None,
    backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
    mixed_precision: Optional[MixedPrecision] = None,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    device_id: Optional[Union[int, torch.device]] = None,
    sync_module_states: bool = False,
    forward_prefetch: bool = False,
    limit_all_gathers: bool = True,
    use_orig_params: bool = False,
    ignored_states: Union[Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]] = None,
    # HACK Add some new parameters
    device_mesh: Optional[DeviceMesh] = None,
    use_local_sgd: bool = False,
    local_sgd_sync_interval: int = 1,
    local_sgd_warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    outer_optim_class: Optional[Type[Optimizer]] = None,
    outer_optim_kwargs: Optional[dict] = {},
    outer_optim_cpu_offload: Optional[bool] = False,
    # GTA related args
    reducer: Optional[Literal["linear", "gta"]] = None,
    consensus_method: Optional[Literal["sum", "count"]] = None,
    sparsification_method: Optional[Literal["magnitude", "random", "rescaled_random"]] = None,
    normalize: bool = True,
    density: float = 1.0,
    int8_mask: bool = False,
):
    torch._C._log_api_usage_once("torch.distributed.fsdp")
    super(type(self), self).__init__()
    # HACK Initializes local sgd related process group (if device_mesh is provided)
    if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
        # Version=2.1.0 doesn't support device_mesh, so I modify here.
        if process_group is None and device_mesh is not None:
            if _is_valid_hybrid_shard_device_mesh(device_mesh):
                process_group = (device_mesh.get_dim_groups(mesh_dim=1), device_mesh.get_dim_groups(mesh_dim=0))
            else:
                raise ValueError(f"Expected device_mesh to have ndim=2 but got {len(device_mesh.get_dim_groups())}")

    _init_ignored_module_states(self, module, ignored_modules, ignored_states)
    _init_device_handle(self, module, self._ignored_params, device_id)

    # Add module annotations for Dynamo support (see function for details)
    _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

    # Initializes self.process_group, along with rank and world size. This will
    # also set another attribute, _inter_node_pg, to control the process group
    # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
    # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
    # the same process group state as the root FSDP module.
    _init_process_group_state(self, process_group, sharding_strategy, auto_wrap_policy)
    if auto_wrap_policy is not None:
        root_kwargs = {
            "process_group": process_group,
            "sharding_strategy": sharding_strategy,
            "cpu_offload": cpu_offload,
            "backward_prefetch": backward_prefetch,
            "mixed_precision": mixed_precision,
            "param_init_fn": param_init_fn,
            "device_id": device_id,
            "sync_module_states": sync_module_states,
            "forward_prefetch": forward_prefetch,
            "limit_all_gathers": limit_all_gathers,
            "use_orig_params": use_orig_params,
            "ignored_states": self._ignored_params,
            # HACK Add the new parameters to recursively wrap modules
            "device_mesh": device_mesh,
            "use_local_sgd": use_local_sgd,
            "local_sgd_sync_interval": local_sgd_sync_interval,
            "local_sgd_warmup_steps": local_sgd_warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "outer_optim_class": outer_optim_class,
            "outer_optim_kwargs": outer_optim_kwargs,
            "outer_optim_cpu_offload": outer_optim_cpu_offload,
            # HACK add gta reducer args
            "reducer": reducer,
            "consensus_method": consensus_method,
            "sparsification_method": sparsification_method,
            "normalize": normalize,
            "density": density,
            "int8_mask": int8_mask,
        }
        if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            # Share root process groups with children to maintain
            # the invariant that all FSDP modules will have the same
            # process groups.
            root_kwargs["process_group"] = (self.process_group, self._inter_node_pg)

        _auto_wrap(
            module,
            auto_wrap_policy,
            self._ignored_modules,
            self._ignored_params,
            root_kwargs,
            type(self),
        )

    backward_prefetch_limit = 1
    forward_prefetch_limit = 1
    _init_core_state(
        self,
        sharding_strategy,
        mixed_precision,
        cpu_offload,
        limit_all_gathers,
        use_orig_params,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    _init_runtime_state(self)
    _init_prefetching_state(self, backward_prefetch, forward_prefetch)
    _init_buffer_state(self, module)
    _init_param_handle_from_module(
        self,
        module,
        device_id,
        param_init_fn,
        sync_module_states,
    )
    self._fsdp_wrapped_module = module
    if not use_orig_params:
        _check_orig_params_flattened(self, self._ignored_params)
        _register_flat_param(self, self)

    # HACK Initializes local sgd related states
    _init_local_sgd_state(
        self,
        sharding_strategy,
        use_local_sgd,
        local_sgd_sync_interval,
        local_sgd_warmup_steps,
        gradient_accumulation_steps,
        outer_optim_class,
        outer_optim_kwargs,
        outer_optim_cpu_offload,
        # GTA related args
        reducer,
        consensus_method,
        sparsification_method,
        normalize,
        density,
        int8_mask,
    )

    # `_state_dict_type` controls the `state_dict()` behavior, which is
    # implemented using post-save and pre-load hooks
    _init_state_dict_state(self)
    _register_all_state_dict_hooks(self)


@no_type_check
def _init_param_handle_from_module(
    state: _FSDPState,
    fully_sharded_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _FSDPState:
    """
    Initializes a ``FlatParamHandle`` from a module ``fully_sharded_module``.
    """
    _check_single_device_module(fully_sharded_module, state._ignored_params, device_id)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(
        fully_sharded_module, state._ignored_params, state._ignored_modules
    )
    # Materialize the module if needed
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _materialize_with_param_init_fn(fully_sharded_module, param_init_fn)
    elif is_meta_module:
        _materialize_meta_module(fully_sharded_module, device_id)
    elif is_torchdistX_deferred_init:
        deferred_init.materialize_module(
            fully_sharded_module,
            check_fn=lambda k: _get_module_fsdp_state(k) is None,
        )
    _move_module_to_device(fully_sharded_module, state._ignored_params, device_from_device_id)
    state.compute_device = _get_compute_device(
        fully_sharded_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )

    managed_params = list(_get_orig_params(fully_sharded_module, state._ignored_params))
    if sync_module_states:
        _sync_module_params_and_buffers(fully_sharded_module, managed_params, state.process_group)
        # HACK Modify from the newest FSDP for sync modules in _inter_node_pg.
        if state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            _sync_module_params_and_buffers(fully_sharded_module, managed_params, state._inter_node_pg)
    _init_param_handle_from_params(state, managed_params, fully_sharded_module)
    return state


# HACK Add `_init_local_sgd_state` to initialize local sgd related states.
@no_type_check
def _init_local_sgd_state(
    state: _FSDPState,
    sharding_strategy: Optional[ShardingStrategy] = None,
    use_local_sgd: bool = False,
    local_sgd_sync_interval: int = 1,
    local_sgd_warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    outer_optim_class: Optional[Type[Optimizer]] = None,
    outer_optim_kwargs: Optional[dict] = {},
    outer_optim_cpu_offload: Optional[bool] = False,
    # GTA related args
    reducer: Optional[Literal["linear", "gta"]] = None,
    consensus_method: Optional[Literal["sum", "count"]] = None,
    sparsification_method: Optional[Literal["magnitude", "random", "rescaled_random"]] = None,
    normalize: bool = True,
    density: float = 1.0,
    int8_mask: bool = False,
) -> _FSDPState:
    if use_local_sgd and (sharding_strategy not in HYBRID_SHARDING_STRATEGIES):
        raise RuntimeError("Local SGD only supports hybrid sharding strategies.")
    if use_local_sgd:
        if local_sgd_sync_interval < 1:
            raise ValueError("Invalid local_sgd_sync_interval value: {}.".format(local_sgd_sync_interval))
        if local_sgd_warmup_steps < 0:
            raise ValueError("Invalid local_sgd_warmup_steps value: {}.".format(local_sgd_warmup_steps))
        if gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps value: {}.".format(gradient_accumulation_steps))
    state.use_local_sgd = use_local_sgd
    state.local_sgd_sync_interval = local_sgd_sync_interval
    state.local_sgd_warmup_steps = local_sgd_warmup_steps
    state.gradient_accumulation_steps = gradient_accumulation_steps
    state.global_step = 0
    state.temp_step = 0

    state.outer_optim_class = outer_optim_class
    state.outer_optim_cpu_offload = outer_optim_cpu_offload
    state.last_synced_params = None
    # lazy init outer optimizer
    if use_local_sgd and outer_optim_class is not None:
        state.use_outer_optim = True
        state.outer_optim_kwargs = outer_optim_kwargs
        state.outer_optimizer = None
    else:
        state.use_outer_optim = False

    if reducer == "linear":
        state.gta_reducer = LinearReducer(process_group=state._inter_node_pg, normalize=normalize)
    elif reducer == "gta":
        state.gta_reducer = GTAReducer(
            process_group=state._inter_node_pg,
            consensus_method=consensus_method,
            sparsification_method=sparsification_method,
            normalize=normalize,
            density=density,
            int8_mask=int8_mask,
        )
    else:
        state.gta_reducer = None

    return state


# HACK Modify from the newest FSDP for validing the provided `device_mesh`.
@no_type_check
def _is_valid_hybrid_shard_device_mesh(device_mesh: DeviceMesh) -> bool:
    parent_mesh = mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh is not None:
        raise RuntimeError(
            f"Found device_mesh {device_mesh} passed in has a parent device_mesh {parent_mesh}.",
            "Hybrid sharding + TP is not supported yet.",
        )
    return isinstance(device_mesh, DeviceMesh) and device_mesh.ndim == 2
