# MODIFIED from torch.distributed.fsdp._state_dict_utils
import contextlib
import os
import warnings
from typing import Any, Dict, Optional, cast, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _module_handle,
)
from torch.distributed.fsdp._state_dict_utils import (
    _full_post_state_dict_hook,
    _full_pre_state_dict_hook,
    _local_post_state_dict_hook,
    _local_pre_state_dict_hook,
    _replace_with_full_state_dict_type,
    _sharded_post_state_dict_hook,
    _sharded_pre_state_dict_hook,
    logger,
)
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

from ._runtime_utils import _lazy_init_outer_optimizer, _sync_sharded_params


# HACK Synchronize parameters once before returning model state_dict.
@no_type_check
def _pre_state_dict_sync(
    module: nn.Module,
    state: _FSDPState,
) -> None:
    """
    Synchronization of model state dict is divided into two stages:
    1. If it is in the synchronous training phase, no synchronization is required and this func directly returns;
    2. If it is in the asynchronous training phase,
       the model parameters are synchronized and the state dict of the outer optimizer is updated at the same time.
    """
    # It is still in sync stage.
    if not (state.use_local_sgd and state.global_step >= state.local_sgd_warmup_steps + 1):
        return

    state._device_handle.synchronize()
    handle = _module_handle(state, module)

    _sync_sharded_params(state, handle)

    if state.local_sgd_cpu_offload and (state.gta_reducer is not None or state.use_outer_optim):
        state._H2D_stream.synchronize()
        state._D2H_stream.synchronize()
    state._average_stream.synchronize()
    dist.barrier()


@no_type_check
def _pre_state_dict_hook(
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    This is called before the core state dict saving logic of ``module``.
    ``fsdp_state._state_dict_type`` is used to decide what postprocessing will
    be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn("When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will" "be returned.")
    else:
        _set_use_dtensor(fsdp_state)
        context = contextlib.nullcontext()

    with context:
        # HACK Synchronize parameters once before returning model state_dict.
        _pre_state_dict_sync(module, fsdp_state)

        _pre_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_state_dict_hook,
        }
        _pre_state_dict_hook_fn[fsdp_state._state_dict_type](
            fsdp_state,
            module,
            *args,
            **kwargs,
        )


@no_type_check
@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``fsdp_state._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn("When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will" "be returned.")
    else:
        context = contextlib.nullcontext()

    with context:
        _post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
        }
        processed_state_dict = _post_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )

    # HACK remove the `last_synced_params` related state.
    if fsdp_state.use_local_sgd:
        keys_to_remove = [key for key, _ in sorted(processed_state_dict.items()) if "last_synced_params" in key]
        for key in keys_to_remove:
            del processed_state_dict[key]

    if fsdp_state._is_root:
        logger.info("FSDP finished processing state_dict(), prefix=%s", prefix)
        for key, tensor in sorted(processed_state_dict.items()):
            if key.startswith(prefix) and isinstance(tensor, torch.Tensor):
                local_shape = tensor.shape
                if isinstance(tensor, ShardedTensor):
                    local_shape = None
                    shards = tensor.local_shards()
                    if shards:
                        local_shape = shards[0].tensor.shape
                elif isinstance(tensor, DTensor):
                    local_shape = tensor.to_local().shape
                logger.info(
                    "FQN=%s: type=%s, shape=%s, local_shape=%s, dtype=%s, device=%s",
                    key,
                    type(tensor),
                    tensor.shape,
                    local_shape,
                    tensor.dtype,
                    tensor.device,
                )

    return processed_state_dict


@no_type_check
def _set_use_dtensor(fsdp_state: _FSDPState) -> None:
    # If device_mesh is passed in when initalizing FSDP, we automatically turn the
    # _use_dtensor flag to be true for ShardedStateDictConfig().
    if getattr(fsdp_state, "_device_mesh", None):
        state_dict_type = fsdp_state._state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError(
                "Found state_dict_type LOCAL_STATE_DICT",
                "DeviceMesh is not compatible with LOCAL_STATE_DICT.",
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.",
            )
        elif state_dict_type == StateDictType.FULL_STATE_DICT:
            logger.warning(
                "Found both state_dict_type FULL_STATE_DICT and device_mesh. "  # noqa: G004
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict."
            )
        else:
            fsdp_state._state_dict_config._use_dtensor = True


# HACK Collect the local sgd related state-dicts.
@no_type_check
def _local_sgd_state_dict(
    model: torch.nn.Module,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Transform the local sgd related state-dicts corresponding to a sharded model.

    The given state-dict can be transformed to one of two types to be consistent with normal state-dicts:
    1) full state_dict, 2) sharded/local state_dict.

    For full state_dict, all states remain flattened and sharded while will be gathered in a list.
    Rank0 only and CPU only can be specified via :meth:`state_dict_type` to avoid OOM.

    For sharded/local state_dict, all states are also flattened and sharded.
    CPU only can be specified via :meth:`state_dict_type` to further save memory.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.distributed.fsdp import StateDictType
        >>> from torch.distributed.fsdp import FullStateDictConfig
        >>> from torch.distributed.fsdp import FullOptimStateDictConfig
        >>> # Save a checkpoint
        >>> model = ...
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> state_dict = model.state_dict()
        >>> local_sgd_state_dict = FSDP.local_sgd_state_dict(model)
        >>> save_a_checkpoint(state_dict, local_sgd_state_dict)
        >>> # Load a checkpoint
        >>> model = ...
        >>> state_dict = load_a_checkpoint()
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> model.load_state_dict(state_dict)
        >>> FSDP.load_local_sgd_state_dict(model, load_dir = "./local_sgd_ckpt")

    Args:
        model (torch.nn.Module): Root module.
        group (dist.ProcessGroup): Model's process group across which parameters
            are sharded or ``None`` if using the default process group. (
            Default: ``None``)

    Returns:
        Dict[str, Any]: A :class:`dict` containing the local sgd state for
        ``model``. The sharding of the local sgd state is based on
        ``state_dict_type``.
    """
    if not model.use_local_sgd or model.global_step < model.local_sgd_warmup_steps:
        return {}

    state_dict_settings = torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type(model)
    local_sgd_sd = {"global_step": model.global_step}
    rank0_only = (
        state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT
        and cast(FullStateDictConfig, state_dict_settings.state_dict_config).rank0_only
    )
    cpu_offload = getattr(state_dict_settings.optim_state_dict_config, "offload_to_cpu", True)

    # Only collect one copy of the local sgd state dict in all nodes.
    if dist.get_rank(model._inter_node_pg) != 0:
        return {}

    if not model.use_outer_optim or model.outer_optimizer is None:
        if rank0_only and model.rank != 0:
            return {}
        return local_sgd_sd

    fsdp_modules = torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules(model)
    outer_optim_sd_l = []

    for fsdp_m in fsdp_modules:
        if fsdp_m.outer_optimizer is None:
            fsdp_m_outer_optim_sd = {}
        else:
            fsdp_m_outer_optim_sd = fsdp_m.outer_optimizer.state_dict()
            fsdp_m_outer_optim_sd["rank"] = fsdp_m.rank
            if cpu_offload:
                for outer_optim_state in fsdp_m_outer_optim_sd["state"].values():
                    for k, v in outer_optim_state.items():
                        if torch.is_tensor(v):
                            outer_optim_state[k] = v.to("cpu")
        outer_optim_sd_l.append(fsdp_m_outer_optim_sd)

    if group is None:
        group = model.process_group
    if state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT:
        if rank0_only:
            gathered_outer_optim_sd_l = [None] * model.world_size if model.rank == 0 else None
            dist.gather_object(
                outer_optim_sd_l,
                object_gather_list=gathered_outer_optim_sd_l,
                dst=dist.get_rank() - model.rank,
                group=group,
            )
            if model.rank == 0:
                local_sgd_sd["outer_optim"] = gathered_outer_optim_sd_l
            else:
                local_sgd_sd = {}
        else:
            gathered_outer_optim_sd_l = [None] * model.world_size
            dist.all_gather_object(object_list=gathered_outer_optim_sd_l, obj=outer_optim_sd_l, group=group)
            local_sgd_sd["outer_optim"] = gathered_outer_optim_sd_l
    else:
        local_sgd_sd["outer_optim"] = outer_optim_sd_l

    return local_sgd_sd


# HACK Save the local sgd related state-dicts.
@no_type_check
def _save_local_sgd_state_dict(
    model: torch.nn.Module,
    group: Optional[dist.ProcessGroup] = None,
    rank0_only: bool = True,
    full_state_dict: bool = True,
    cpu_offload: bool = True,
    save_dir: str = "./local_sgd_ckpt",
    ckpt_name: str = "local_sgd_ckpt",
) -> None:
    """
    Save the collected local sgd related state-dicts corresponding to a sharded model.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.distributed.fsdp import StateDictType
        >>> from torch.distributed.fsdp import FullStateDictConfig
        >>> from torch.distributed.fsdp import FullOptimStateDictConfig
        >>> # Save a checkpoint
        >>> model = ...
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> FSDP.save_local_sgd_state_dict(model, save_dir = "./ckpt")
        >>> # Load a checkpoint
        >>> model = ...
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> FSDP.load_local_sgd_state_dict(model, load_dir = "./ckpt")

    Args:
        model (torch.nn.Module): Root module.
        group (dist.ProcessGroup): Model's process group across which parameters
            are sharded or ``None`` if using the default process group. (
            Default: ``None``)
        rank0_only (bool): If ``True``, collects the populated :class:`dict`
            only on rank 0; if ``False``, collects it on all ranks. (Default: ``True``)
        full_state_dict (bool): If ``True``, uses ``FULL_STATE_DICT``; if ``False``, uses
            ``SHARDED_STATE_DICT``. (Default: ``True``)
        cpu_offload (bool): If ``True``, offloads the state dict to CPU; if ``False``, the
            devices will not be modified. (Default: ``True``)
        save_dir (str): Save directory. (Default: ``"./local_sgd_ckpt"``)
        ckpt_name (str): Checkpoint file name. (Default: ``".local_sgd_ckpt"``)

    Returns:
        None.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    state_dict_context = contextlib.nullcontext()
    if full_state_dict:
        state_dict_context = torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=cpu_offload, rank0_only=rank0_only),
            FullOptimStateDictConfig(offload_to_cpu=cpu_offload, rank0_only=rank0_only),
        )
    else:
        state_dict_context = torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=cpu_offload),
            ShardedOptimStateDictConfig(offload_to_cpu=cpu_offload),
        )

    with state_dict_context:
        local_sgd_state_dict = _local_sgd_state_dict(model, group)
        local_sgd_state_dict = torch.utils._pytree.tree_map(
            lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, local_sgd_state_dict
        )
        dp_rank = dist.get_rank()

        if full_state_dict:
            if dp_rank == 0:
                if local_sgd_state_dict:
                    local_sgd_save_full_path = os.path.join(save_dir, f"{ckpt_name}.pt")
                    torch.save(local_sgd_state_dict, local_sgd_save_full_path)
                    logger.info(
                        "Rank [%s] --> full local sgd checkpoint saved at %s",
                        dp_rank,
                        local_sgd_save_full_path,
                    )
        else:
            if local_sgd_state_dict:
                local_sgd_save_full_path = os.path.join(save_dir, f"{ckpt_name}_{dp_rank:02d}.pt")
                torch.save(local_sgd_state_dict, local_sgd_save_full_path)
                logger.info(
                    "Rank [%s] --> sharded local sgd checkpoint saved at %s",
                    dp_rank,
                    local_sgd_save_full_path,
                )


# HACK Load the local sgd related state-dicts.
@no_type_check
def _load_local_sgd_state_dict(
    model: torch.nn.Module,
    group: Optional[dist.ProcessGroup] = None,
    rank0_only: bool = True,
    full_state_dict: bool = True,
    load_dir: str = "./local_sgd_ckpt",
    ckpt_name: str = "local_sgd_ckpt",
) -> None:
    """
    Load the collected local sgd related state-dicts corresponding to a sharded model.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.distributed.fsdp import StateDictType
        >>> from torch.distributed.fsdp import FullStateDictConfig
        >>> from torch.distributed.fsdp import FullOptimStateDictConfig
        >>> # Save a checkpoint
        >>> model = ...
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> FSDP.save_local_sgd_state_dict(model, save_dir = "./ckpt")
        >>> # Load a checkpoint
        >>> model = ...
        >>> FSDP.set_state_dict_type(
        >>>     model,
        >>>     StateDictType.FULL_STATE_DICT,
        >>>     FullStateDictConfig(rank0_only=False),
        >>>     FullOptimStateDictConfig(rank0_only=False),
        >>> )
        >>> FSDP.load_local_sgd_state_dict(model, load_dir = "./ckpt")

    Args:
        model (torch.nn.Module): Root module.
        group (dist.ProcessGroup): Model's process group across which parameters
            are sharded or ``None`` if using the default process group. (
            Default: ``None``)
        rank0_only (bool): If ``True``, collects the populated :class:`dict`
            only on rank 0; if ``False``, collects it on all ranks. (Default: ``True``)
        full_state_dict (bool): If ``True``, uses ``FULL_STATE_DICT``; if ``False``, uses
            ``SHARDED_STATE_DICT``. (Default: ``True``)
        load_dir (str): The directory where the state dict are saved. (Default: ``"./local_sgd_ckpt"``)
        ckpt_name (str): Checkpoint file name. (Default: ``".local_sgd_ckpt"``)

    Returns:
        None.
    """

    if not model.use_local_sgd:
        return

    local_sgd_sd = None
    if full_state_dict:
        local_sgd_load_full_path = os.path.join(load_dir, f"{ckpt_name}.pt")
        if not os.path.exists(local_sgd_load_full_path):
            raise FileNotFoundError(f"The checkpoint file: {local_sgd_load_full_path} does not exist!")
        if rank0_only:
            if model.rank == 0:
                all_local_sgd_sd = torch.load(local_sgd_load_full_path)
                if "outer_optim" not in all_local_sgd_sd.keys():
                    all_local_sgd_sd = [{"global_step": all_local_sgd_sd["global_step"]}] * model.world_size
                else:
                    all_local_sgd_sd = [
                        {"global_step": all_local_sgd_sd["global_step"], "outer_optim": outer_optim_rank}
                        for outer_optim_rank in all_local_sgd_sd["outer_optim"]
                    ]
            else:
                all_local_sgd_sd = [None] * model.world_size
            local_sgd_sd = [None]
            if group is None:
                group = model.process_group
            dist.scatter_object_list(
                local_sgd_sd,
                all_local_sgd_sd,
                src=dist.get_rank() - model.rank,
                group=group,
            )
            local_sgd_sd = local_sgd_sd[0]
        else:
            all_local_sgd_sd = torch.load(local_sgd_load_full_path)
            if "outer_optim" not in all_local_sgd_sd.keys():
                local_sgd_sd = all_local_sgd_sd
            else:
                local_sgd_sd = {
                    "global_step": all_local_sgd_sd["global_step"],
                    "outer_optim": all_local_sgd_sd["outer_optim"][model.rank],
                }
    else:
        local_sgd_load_full_path = os.path.join(load_dir, f"{ckpt_name}_{model.rank:02d}.pt")
        if not os.path.exists(local_sgd_load_full_path):
            raise FileNotFoundError(f"The checkpoint file: {local_sgd_load_full_path} does not exist!")
        local_sgd_sd = torch.load(local_sgd_load_full_path)

    fsdp_modules = torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules(model)

    for i, fsdp_m in enumerate(fsdp_modules):
        fsdp_m.global_step = local_sgd_sd["global_step"]
        if "outer_optim" in local_sgd_sd.keys():
            _lazy_init_outer_optimizer(fsdp_m, fsdp_m._handle, cpu_init=fsdp_m.local_sgd_cpu_offload)
            assert (
                fsdp_m.rank == local_sgd_sd["outer_optim"][i]["rank"]
            ), "The rank of FSDP module and state dict do not match."
            del local_sgd_sd["outer_optim"][i]["rank"]
            fsdp_m.outer_optimizer.load_state_dict(local_sgd_sd["outer_optim"][i])

    logger.info("Rank [%s] --> local sgd checkpoint loaded!", dist.get_rank())
