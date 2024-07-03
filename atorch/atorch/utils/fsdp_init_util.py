"""This file is used to restore FlatParameter from flat ckpt."""
import collections
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union, no_type_check

import torch
from torch import nn
from torch.distributed.fsdp import _init_utils, flat_param
from torch.distributed.fsdp import fully_sharded_data_parallel as FSDP
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.utils import _p_assert

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import local_rank, rank, world_size
from atorch.utils.fsdp_save_util import _FLAT_PARAM_PADDING_VALUE, ErrorCode, FlatCkptError, ShardTensorUtil, TensorDict
from atorch.utils.meta_model_utils import _find_tied_weights, _retie_weights

OriginFlatParamHandle: Type[flat_param.FlatParamHandle] = flat_param.FlatParamHandle
origin_init_param_handle_from_module = _init_utils._init_param_handle_from_module


def patch_fsdp_init(ckpt_path, wrap_class, top_model, check_module=True, ignore_ckpt_version=False):
    """This func will patch 2 functions/classes:
        1. FSDP._init_param_handle_from_module
        2. _init_utils.FlatParamHandle

    wrap_class: Sequence[type], sequence of classes.
    top_model: nn.Module, top level module.
    check_module: bool, check class module or not, maybe same class name but different module name.
    ignore_ckpt_version: boot, flat ckpt version, current support 2 versions in [0,1]

    It is dangerous to patch core functions in FSDP, so we only support pt2.1.0.
    """
    version = torch.version.git_version
    if version == "7bcf7da3a268b435777fe87c7794c382f444e86d":
        RestoreFlatParamHandle.GLOBAL_CONFIG.build_ckpt_util(ckpt_path, wrap_class, top_model)
        if not check_module:
            logger.warn("Ignore module checkint, make sure your wrap class in FSDP is same")
        elif not RestoreFlatParamHandle.GLOBAL_CONFIG.compare_wrap_class(ignore_ckpt_version=ignore_ckpt_version):
            ckpt_wrap_class: List[Tuple[str, str]] = list(
                RestoreFlatParamHandle.GLOBAL_CONFIG.ckpt_util.ckpt_meta["wrap_class"]
            )
            raise FlatCkptError(
                f"wrap class mismatch ckpt({ckpt_wrap_class}) vs model({wrap_class}",
                ErrorCode.CHECKPOINT_WRAP_CLASS_MISMATCH,
            )

        RestoreFlatParamHandle.shard = RestoreFlatParamHandle.mock_shard_pt210
        FSDP._init_param_handle_from_module = mock_init_param_handle_from_module_pt210
        _init_utils.FlatParamHandle = RestoreFlatParamHandle
    else:
        raise FlatCkptError(
            "Torch git version must equal 7bcf7da3a268b435777fe87c7794c382f444e86d(2.1.0)", ErrorCode.NOT_SUPPORT
        )


def reset_hooks():
    global _init_utils, FSDP
    _init_utils.FlatParamHandle = OriginFlatParamHandle
    FSDP._init_param_handle_from_module = origin_init_param_handle_from_module


@dataclass
class RestoreFlatParamHandleInjectConfig:
    """We inherit `FlatParamHandle` from FSDP, as we don't want to change any function signature of
    origin `FlatParamHandle`, so we use this to inject for configuring RestoreFlatParamHandle.
    All `RestoreFlatParamHandle` objects use same instance of `RestoreFlatParamHandleInjectConfig`.

    ckpt_path: flat ckpt path
    wrap_class: wrap class for FSDP, if wrap class has changed, restore will failed.
    top_model: origin module at top level.
    """

    ckpt_path: Optional[str] = None
    wrap_class: Optional[Sequence[type]] = None
    top_model: Optional[torch.nn.Module] = None

    def build_ckpt_util(self, ckpt_path, wrap_class, top_model):
        """This will build util of flat ckpt and get all init order of FSDP."""
        self.ckpt_path = ckpt_path
        self.wrap_class = set(wrap_class) if isinstance(wrap_class, collections.Sequence) else {wrap_class}
        self.top_model = top_model
        self.ckpt_util = ShardTensorUtil(self.ckpt_path, rank(), world_size(), device="cpu")
        self.ckpt_util.get_fsdp_init_order(self.top_model, self.wrap_class, build_fsdp_load_map=False)

    def compare_wrap_class(self, ignore_ckpt_version=False):
        assert self.wrap_class is not None  # mypy
        wrap_class_config: List[Tuple[str, str]] = [(i.__module__, i.__name__) for i in self.wrap_class]
        ckpt_wrap_class: List[Tuple[str, str]] = list(self.ckpt_util.ckpt_meta["wrap_class"])

        if self.ckpt_util.ckpt_meta["version"] == 0:
            logger.warn("Meet old flat ckpt, make sure your wrap class in FSDP is same")
            return ignore_ckpt_version
        wrap_class_config = sorted(wrap_class_config, key=lambda x: ".".join(x))
        ckpt_wrap_class = sorted(ckpt_wrap_class, key=lambda x: ".".join(x))
        return wrap_class_config == ckpt_wrap_class

    def enable(self):
        """Check enable of not"""
        return all(i is not None for i in vars(self).values())


class RestoreFlatParamHandle(OriginFlatParamHandle):  # type: ignore
    """We restore sharded FlatParameter, we do this in `shard` function, load shared flat param in ckpt.
    FSDP use shard views in forward/backward pass, so FSDP must init params in CUDA devices, if not,
    all views assignemt will raise.

    Origin load will let rank0 to load all params/buffers in ckpt, which use mmap(from safetensors),
    this will cause many tlb miss to hurt performance. Then broadcast to all ranks.

    After optimize, each rank load flat_param which belones it self, maybe need to reshard if world size changed.

    We use ascii to illustrate it.
                                       ORIGIN INIT

                                                    2.broadcast
                           +-------------------------+-------------------+
                           |                         |                   |
                           |                         v                   v
                      +----+----+               +----+----+         +----+----+
                      |         |               |         |         |         |
                      |  rank0  |               |  rank1  |         | rank..n |
                      |         |               |         |         |         |
                      +----+--+-+               +----+----+         +----+----+
                           |  |                      |                   |
                           |  |                      |                   |           +----------------------+
                           |  +----------------------+-------------------+---------->+                      |
                           |                     3.create handle                     |  FlatParameterHandle |
                           |                                                         |                      |
                           |   1.load and concat/reshape                             +----------------------+
                           +-----------+
                           |           |
                           v           v
           +---------------+---+-------+--------------+
           |     flat_param_0  |    flat param 1      |
           +-------------------+----------------------+


     ============================================================================================================

                                       OPTIMIZED INIT

                      +---------+                            +---------+         +---------+
                      |         |                            |         |         |         |
                      |  rank0  |                            |  rank1  |         | rank..n |
                      |         |                            |         |         |         |
                      +----+----+                            +---------+         +---------+
                           | 1.get sharded flat_param
                           +-----------+                    same as rank0       same as rank0
                           |           |
                           v           v
           +---------------+---+-------+--------------+
           |     flat_param_0  |    flat param 1      |
           +-------------------+----------------------+

    The whole restore steps is:
        1. build model on meta device
        2. find tie weights
        2. use `param_init_fn` to materialize_module, `param_init_fn` can be
            lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        3. retie weights
        4. flat tensors to construct origin flat_param in FlatParamHandle.
            This can be optimized out, but this doing on gpu, it's fast.
        5. load reshard flat param in ckpt, and set data pointer of flat_param
        6. FSDP don't manage buffers, so we reload buffers.

    """

    GLOBAL_CONFIG = RestoreFlatParamHandleInjectConfig()
    RESTORE_FLAT_PARAM_HANDLE_TAG = "ATORCH_FSDP_INIT_ORDER"
    HAS_TAGGING = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable = RestoreFlatParamHandle.GLOBAL_CONFIG.enable()
        self.post_init()

    def post_init(self):
        """Tagging name for each FSDP unit, which name is flat_param's name in flat ckpt. After this,
        we can load flat_param in ckpt for this FSDP unit.
        More decsription of init_module_names is in fsdp_save_util.py:get_fsdp_init_order.
        """
        if not self.enable:
            return

        self.ckpt_util = RestoreFlatParamHandle.GLOBAL_CONFIG.ckpt_util
        if RestoreFlatParamHandle.HAS_TAGGING:
            return
        for name, module in RestoreFlatParamHandle.GLOBAL_CONFIG.top_model.named_modules():
            if name in RestoreFlatParamHandle.GLOBAL_CONFIG.ckpt_util.init_module_names:
                setattr(module, RestoreFlatParamHandle.RESTORE_FLAT_PARAM_HANDLE_TAG, name)
        RestoreFlatParamHandle.HAS_TAGGING = True

    @torch.no_grad()
    def mock_shard_pt210(self):
        """Mock origin `shard` function.
        We load shared flat param from ckpt and set it to flat_param in `FlatParamHandle`.
        """

        def load_buffer():
            """Reload buffers."""
            module = self._fully_sharded_module
            module_name = self._fully_sharded_module.ATORCH_FSDP_INIT_ORDER
            for name, buf in module.named_buffers():
                name = clean_tensor_name(name)
                buffer_name_in_ckpt = f"{module_name}.{name}" if module_name else name
                ckpt_buf = self.ckpt_util.buffers.get_tensor(buffer_name_in_ckpt)
                ckpt_buf = ckpt_buf.to(buf.device)
                ckpt_buf = ckpt_buf.to(buf.dtype)
                buf.set_(ckpt_buf)

        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            orig_storage = flat_param._typed_storage()
            flat_param_name_in_ckpt = getattr(
                self._fully_sharded_module, RestoreFlatParamHandle.RESTORE_FLAT_PARAM_HANDLE_TAG, None
            )
            if flat_param_name_in_ckpt is None:
                raise ValueError("Ckpt mismatch, maybe you have change wrapclass or number of layer")
            sharded_flat_param, numel_padded = self.ckpt_util.load_flat_param_by_name(
                self._fully_sharded_module.ATORCH_FSDP_INIT_ORDER, self
            )
            flat_param.set_(
                sharded_flat_param.to(torch.device("cuda", index=local_rank()))
            )  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
            # load buffer here
            load_buffer()
            if orig_storage._size() > 0:
                orig_storage._resize_(0)
        if self._use_orig_params:
            self._use_sharded_views()


def _check_weight_is_init(module_name: str, module: nn.Module) -> None:
    """
    Check first param is 42 or not.
    """
    for name, p in module.named_parameters():
        if p.view(-1)[0] == _FLAT_PARAM_PADDING_VALUE:
            raise FlatCkptError(f"lora weight {module_name}.{name} is not init", ErrorCode.LORA_WEIGHT_INIT_ERROR)


def _reset_lora_param(module: nn.Module, lora_cls: Tuple[type]) -> None:
    """
    Reset all lora params before FSDP init, this function is called in `FSDP._init_param_handle_from_module`.
    This function is called after `FSDP.param_init_fn`.

    It's useful when base model and lora model are init on meta device. We can load base model from ckpt, but
    lora weights are maybe training from scratch, at this time, we do not know how them initialize. Peft offers
    reset function in lora, but other lora method are not tested.
    """

    for name, m in module.named_modules():
        if isinstance(m, lora_cls):
            try:
                m.reset_lora_parameters(m.active_adapter)  # type: ignore[attr-defined]
            except Exception as e:
                import peft

                raise FlatCkptError(
                    f"Reset lora weights error, your version of peft is {peft.__version}. Check the origin call stack",
                    ErrorCode.LORA_RESET_WEIGHT_ERROR,
                ) from e
            _check_weight_is_init(name, m)


@no_type_check
def mock_init_param_handle_from_module_pt210(
    state: _init_utils._FSDPState,
    fully_sharded_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
    **atorch_hook_kwargs,
) -> _init_utils._FSDPState:
    """
    We two things, others are same with original function.
        1. add tie_weights after `materialize_meta_module`.
        2. add lora reset parameters
    """
    tie_weights = {}
    tie_weights = _find_tied_weights(fully_sharded_module)

    _init_utils._check_single_device_module(fully_sharded_module, state._ignored_params, device_id)
    device_from_device_id = _init_utils._get_device_from_device_id(device_id, state.rank)
    is_meta_module, is_torchdistX_deferred_init = _init_utils._need_to_materialize_module(
        fully_sharded_module, state._ignored_params, state._ignored_modules
    )
    # Materialize the module if needed
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _init_utils._materialize_with_param_init_fn(fully_sharded_module, param_init_fn)
    elif is_meta_module:
        _init_utils._materialize_meta_module(fully_sharded_module, device_id)
    elif is_torchdistX_deferred_init:
        _init_utils.deferred_init.materialize_module(
            fully_sharded_module,
            check_fn=lambda k: _init_utils._get_module_fsdp_state(k) is None,
        )
    _init_utils._move_module_to_device(fully_sharded_module, state._ignored_params, device_from_device_id)
    _retie_weights(fully_sharded_module, tie_weights)
    state.compute_device = _init_utils._get_compute_device(
        fully_sharded_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )

    wrap_cls = atorch_hook_kwargs.get("atorch_wrap_cls", int)
    restore_lora = atorch_hook_kwargs.get("restore_lora", False)

    if not restore_lora and isinstance(fully_sharded_module, wrap_cls):
        # wrap cls fsdp unit
        lora_cls = atorch_hook_kwargs.get("lora_cls", None)
        _reset_lora_param(fully_sharded_module, lora_cls)

    managed_params = list(_init_utils._get_orig_params(fully_sharded_module, state._ignored_params))

    if "atorch_wrap_cls" in atorch_hook_kwargs:
        # check sync_module_states, and any param is 42
        if not sync_module_states:
            raise ValueError("FSDP lora on meta init must set `sync_module_states`", ErrorCode.LORA_WEIGHT_INIT_ERROR)
        for p in managed_params:
            if p.view(-1)[0] != _FLAT_PARAM_PADDING_VALUE:
                continue
            for name, param in fully_sharded_module.named_parameters():
                if param is p:
                    raise FlatCkptError(f"Weight {name} is not init", ErrorCode.LORA_WEIGHT_INIT_ERROR)
    if sync_module_states:
        _init_utils._sync_module_params_and_buffers(fully_sharded_module, managed_params, state.process_group)
    _init_utils._init_param_handle_from_params(state, managed_params, fully_sharded_module)
    return state


@dataclass
class FSDPCkptConfig:
    """
    This is use for meta init and use `param_init_fn` of FSDP to init params.
    This object has three str:
        1. flat_ckpt_path, dir of flat ckpt
        2. lora_ckpt_path, dir of lora ckpt
        3. lora_prefix, prefix of lora, 'base_model.model' as usually.
        4. lora_cls, lora linear class, defaults to peft.tuners.lorr.Linear

    """

    flat_ckpt_path: Optional[str] = None
    lora_ckpt_path: Optional[str] = None
    lora_prefix: Optional[str] = None
    lora_cls: Optional[List[type]] = None
    lora_weight_name: Optional[str] = None


class FSDPInitFn:
    """
    param_init_fn of FSDP.
    """

    def __init__(self, top_model: nn.Module, rank: int, fsdp_ckpt_config: FSDPCkptConfig, wrap_cls: Tuple[type]):
        if not isinstance(wrap_cls, tuple):
            raise FlatCkptError("kwarg `wrap_clas` of FSDPInitFn must be tuple of class", ErrorCode.COMMON_ERROR)
        self._set_global_name_for_params_and_buffers(top_model)
        self.rank = rank
        self.fsdp_ckpt_config = fsdp_ckpt_config
        self._prepare_ckpt()
        self.wrap_cls = wrap_cls
        self.restore_lora = fsdp_ckpt_config.lora_ckpt_path is not None

        if rank == 0:
            FSDP._init_param_handle_from_module = partial(
                mock_init_param_handle_from_module_pt210,
                atorch_wrap_cls=wrap_cls,
                restore_lora=self.restore_lora,
                lora_cls=self._prepare_lora_cls_check_list(),
            )
        else:
            FSDP._init_param_handle_from_module = mock_init_param_handle_from_module_pt210

    def _prepare_lora_cls_check_list(self):
        """
        We need to call `reset_lora_parameters` from class `LoraLinear`, maybe user will implement `lora` by
        their own, so we pass lora class to filter which object can do `reset_lora_parameters`. defaults to
        `peft.tuners.lorr.Linear`
        """
        from peft.tuners.lora import Linear as LoraLinear

        self.lora_cls = [LoraLinear]
        config_lora_cls = self.fsdp_ckpt_config.lora_cls
        if config_lora_cls is not None:
            self.lora_cls.extend(config_lora_cls)
        self.lora_cls = tuple(self.lora_cls)
        return self.lora_cls

    def _prepare_ckpt(self):
        """
        Parse all ckpts
        """
        flat_ckpt_path = self.fsdp_ckpt_config.flat_ckpt_path
        lora_ckpt_path = self.fsdp_ckpt_config.lora_ckpt_path
        lora_prefix = self.fsdp_ckpt_config.lora_prefix

        # we striped prefix which is created by lora, usually, it's  `base_model.model.`
        self.base_model_weights = ShardTensorUtil(
            flat_ckpt_path,
            0,
            1,
            name_mapping_func_or_dict=lambda x: f"{x[len(lora_prefix)+1:]}" if lora_prefix else x,
            device="cpu",
        )
        self.lora_weights = self._parse_lora_param(lora_ckpt_path)

    def _get_param_buffer_from_flat(self, name) -> Optional[torch.Tensor]:
        """
        Get param or buffer from flat ckpt, if found return tensor, otherwise return None.
        """
        return self.base_model_weights.load_tensor_by_name(name, strict=False)

    def _parse_lora_param(self, path: Optional[str]) -> TensorDict:
        """
        If we set path, it will find `lora_weight` in path.
        If not found, raise error.
        """
        if path is None:
            return {}
        lora_weight_name = self.fsdp_ckpt_config.lora_weight_name or "lora_weight"
        posix_path = Path(path) / lora_weight_name
        if not posix_path.exists():
            raise FlatCkptError(f"lora weights not found, path is {str(posix_path)}", ErrorCode.CHECKPOINT_NOT_FOUND)
        return torch.load(posix_path)

    def _set_global_name_for_params_and_buffers(self, top_model):
        """
        Set extra attr `_atorch_global_name` to record global name of each module.
        It's useful when we iter on submodule.
        """
        for name, param in top_model.named_parameters():
            param._atorch_global_name = name
        for name, buf in top_model.named_buffers():
            buf._atorch_global_name = name

    def __call__(self, module: nn.Module) -> None:
        """
        Real param_init_fn at FSDP's initializing. FSDP must set `sync_module_states`.

        For non root rank, empty init each module, and wait broadcasting from root rank.
        For root rank, there are two cases:
            1.  Lora weights will init from ckpt, in this case, if we not found key in lora_weight dict,
                We raise error.
            2.  Lora weights will init from scratch, in this case, we call `reset_lora_parameters` in peft,
                `wrap_cls` will not be None, so we fill 42 on every param. Before broadcasting, check is any
                param is 42 and raise error.
        """
        if self.rank != 0:
            module.to_empty(device=torch.device("cuda"), recurse=False)
            return
        with torch.no_grad():
            global_names = {name: param._atorch_global_name for name, param in module.named_parameters(recurse=False)}
            global_names.update({name: buf._atorch_global_name for name, buf in module.named_buffers(recurse=False)})
            module.to_empty(device=torch.device("cuda"), recurse=False)
            for name, param in module.named_parameters(recurse=False):
                full_param_name = global_names[name]
                # lookup on flat ckpt
                src_tensor = self._get_param_buffer_from_flat(full_param_name)
                if src_tensor is not None:
                    orig_storage = param._typed_storage()
                    param.set_(src_tensor.cuda().contiguous())
                    if orig_storage._size() > 0:
                        orig_storage._resize_(0)
                # lookup of lora ckpt
                elif full_param_name in self.lora_weights:
                    orig_storage = param._typed_storage()
                    param.set_(self.lora_weights[full_param_name].cuda().contiguous())
                    if orig_storage._size() > 0:
                        orig_storage._resize_(0)
                elif self.restore_lora:
                    raise FlatCkptError(f"Missing lora weight {full_param_name}", ErrorCode.LORA_WEIGHT_INIT_ERROR)
                else:
                    # mark not initialized
                    param.fill_(_FLAT_PARAM_PADDING_VALUE)

            for name, buf in module.named_buffers(recurse=False):
                full_param_name = global_names[name]
                src_tensor = self._get_param_buffer_from_flat(full_param_name)
                # lookup on flat ckpt
                if src_tensor is not None:
                    orig_storage = buf._typed_storage()
                    if orig_storage._size() > 0:
                        orig_storage._resize_(0)
                    buf.set_(src_tensor.contiguous().cuda().to(buf.dtype))
                else:
                    # mark not initialized
                    buf.fill_(_FLAT_PARAM_PADDING_VALUE)
