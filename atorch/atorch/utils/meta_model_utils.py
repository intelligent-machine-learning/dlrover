# Adapted from HuggingFace Accelarate
# These utils handles:
# initialization of a meta model
# offloading to disk
# reloading
import functools
import gc
import os
import shutil
import time
from contextlib import contextmanager
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Dict, Union

import torch

from atorch import local_rank, rank
from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import recursive_setattr
from atorch.utils.fsdp_save_util import ShardTensorUtil
from atorch.utils.fsdp_save_util import version_check as fsdp_save_util_version_check
from atorch.utils.graph_transform_utils import map_aggregate

_super_init_call = torch.nn.Module.__init__


"""
Tie weight:
    model.emb_in.weight = model.emb_out.weight
    This operation will have several effects:
    1. model.emb_out.weight will not be in the named_parameters of the root model
    2. model.emb_out._parameters["weight"] will have its added attributes removed (so our "checkpoint_name") will
        be removed
    So we need to:
    1. restore the checkpoint name
    2. retie the parameters
"""
_TIE_DICT = dict()


class _MetaModeContext:
    offload_path = "./meta_model_offload"
    param_counter = 0
    buffer_counter = 0
    _SHARD_FLAT_PARAM_LOADER: Dict[Union[None, str], ShardTensorUtil] = {}
    current_offload_name = None

    @classmethod
    def del_current_shard_flat_loader(cls):
        if cls.current_offload_name in cls._SHARD_FLAT_PARAM_LOADER:
            del cls._SHARD_FLAT_PARAM_LOADER[cls.current_offload_name]

    @classmethod
    def get_current_shard_flat_loader(cls):
        if not cls._SHARD_FLAT_PARAM_LOADER:
            raise ValueError(
                "Wrong call path, make sure you call `get_current_shard_flat_loader`"
                " after init_empty_weights_with_disk_offload(meta_init_offload_name=xxx)"
            )
        if cls.current_offload_name in cls._SHARD_FLAT_PARAM_LOADER:
            return cls._SHARD_FLAT_PARAM_LOADER[cls.current_offload_name]
        raise ValueError(f"current shard fsdp name is not set, name is {cls.current_offload_name}")


@contextmanager
def load_weights_from_flat_param(ckpt_path, shard_kwargs, meta_init_offload_name):
    fsdp_save_util_version_check()
    if meta_init_offload_name in _MetaModeContext._SHARD_FLAT_PARAM_LOADER:
        raise ValueError(
            f"meta init {meta_init_offload_name} has been create, "
            "check `meta_init_offload_name` in init_empty_weights_with_disk_offload"
        )
    _MetaModeContext._SHARD_FLAT_PARAM_LOADER[meta_init_offload_name] = ShardTensorUtil(
        ckpt_path, rank(), torch.distributed.get_world_size(), device="cpu", **shard_kwargs
    )
    with torch.device("meta"):
        yield


@contextmanager
def init_empty_weights_with_disk_offload(
    ignore_tie_weights=False,
    include_buffers=True,
    reset=True,
    disk_offload=True,
    offload_path=None,
    ckpt_path=None,
    shard_kwargs=None,
    meta_init_offload_name=None,
):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    If distributed process group is initialized, then offloading happens only on rank 0.
    If a model is initialize under this context, it could reloaded with reload_meta_module.

    To make sure all parameters are correctly handled, all methods that
    potentially alter the parameters must be called under this context, e.g. resize_token_embeddings

    TensorParallelOptimization could safely handle modules initialized under this context.

    Transformer based modules oftenly tie decoder/encoder, output/input embedding weights, currently
    this is not properly handled, so by default we do not offload the model

    Args:
        ignore_tie_weights (`bool`, *optional*, defaults to `False`):
            Transformer based modules oftenly tie decoder/encoder, output/input embedding weights.
            Currently, this feature is supported with limited tests. setting ignore_tie_weights to True
            may cause unexpected error but possibly brings efficiency improvement.
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.
        reset (`bool`, *optional*, defaults to `True`): Whether to reset param/buffer counter and the offload disk
        disk_offload (`bool`, *optional*, defaults to `True`): Whether to offload the initialized weights into disk.
            Potential use case: we want to initialize an empty model to hold values to be loaded from customized ckpt
        offload_path (`str`, *optional* defaults to None): Path to which we offload the meta initialized model.
        ckpt_path (`str`, *optional* defaults to None): Checkpoint path for shard fsdp, this will move module to meta
            device and reload param in fsdp transformation.
        shard_kwargs (`dict`,  *optional* defaults to None): Kwargs for shard fsdp util, currently, it has those keys:
            1. name_mapping_func_or_dict: a dict or callable, mapping name of param in restore module to name of param
                in checkpoint.
            2. create_param_cb: callable, if some tensor is missing, use this function to create parameters.

    Example:

    ```python
    import torch.nn as nn
    from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload

    # Initialize a model with 100 billions parameters
    with init_empty_weights_with_disk_offload():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```
    """
    if ckpt_path is not None:
        from atorch.auto.auto_accelerate_context import AutoAccelerateContext

        AutoAccelerateContext.add_ac_attr("FSDP_META_INIT", True)
        with load_weights_from_flat_param(ckpt_path, shard_kwargs or {}, meta_init_offload_name):
            yield
        return
    if offload_path is not None:
        _MetaModeContext.offload_path = offload_path
    if (torch.distributed.is_initialized() and int(local_rank()) == 0) or not torch.distributed.is_initialized():
        if not os.path.isdir(_MetaModeContext.offload_path):
            os.makedirs(_MetaModeContext.offload_path)
        else:
            try:
                if reset:
                    shutil.rmtree(_MetaModeContext.offload_path)
                    os.makedirs(_MetaModeContext.offload_path)
            except OSError as e:
                logger.warning(f"failed to remove offload directory, with error: {e}")
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    if reset:
        _MetaModeContext.param_counter = 0
        _MetaModeContext.buffer_counter = 0

    def register_empty_parameter(module, name, param, *args, **kwargs):
        old_register_parameter(module, name, param)
        if hasattr(param, "checkpoint_name"):
            _TIE_DICT[param] = param.checkpoint_name
        if param is not None:
            # If ckpt name is not set, set it and wait for offloading,
            # otherwise the param is being reset, directly empty the new values
            if not hasattr(module._parameters[name], "checkpoint_name"):
                chk_name = get_checkpoint_name(is_param=True)
                setattr(module._parameters[name], "checkpoint_name", chk_name)
            else:
                pass

    def register_empty_buffer(module, name, buffer, *args, **kwargs):
        old_register_buffer(module, name, buffer)
        if hasattr(buffer, "checkpoint_name"):
            _TIE_DICT[buffer] = buffer.checkpoint_name
        if buffer is not None:
            if not hasattr(module._buffers[name], "checkpoint_name"):
                chk_name = get_checkpoint_name(is_param=False)
                setattr(module._buffers[name], "checkpoint_name", chk_name)
            else:
                pass

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = torch.device("meta")
            return fn(*args, **kwargs)

        return wrapper

    def patch_method(cls, name, new_fn):
        setattr(cls, name, new_fn)

    _reset_call_register = dict()
    _mods_to_clean = list()

    def _hack_reset_param(mod, init_fn, args, kwargs):
        init_res = init_fn(*args, **kwargs)
        if hasattr(mod, "reset_parameters"):
            _orig_reset_call = mod.reset_parameters
            _reset_call_register[mod] = _orig_reset_call

            def reset_empty_parameters(mod, *args, **kwargs):
                reset_res = _orig_reset_call(*args, **kwargs)
                # In case we cannot ignore tie weights,
                # all parameters initialized on local rank 0 will be kept
                # call _set_model_checkpoint_name beforehand to make sure
                # all ranks registers the param names consistently
                _set_model_checkpoint_name(mod)
                if ignore_tie_weights or int(local_rank()) != 0:
                    empty_param(mod, ignore_save=(not disk_offload))
                return reset_res

            setattr(mod, "reset_parameters", lambda *args, **kwargs: reset_empty_parameters(mod, *args, **kwargs))

        return init_res

    @functools.wraps(_super_init_call)
    def init_call_wrapper(mod, *args, **kwargs):
        def init_fn(*args, **kwargs):
            _super_init_call(mod, *args, **kwargs)

        # All models needs to be registered:
        # we need to defer offloading at rank 0 in case tie_weights not ignored
        # the first mod is the root module
        _mods_to_clean.append(mod)

        return _hack_reset_param(mod, init_fn, args, kwargs)

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        patch_method(torch.nn.Module, "__init__", init_call_wrapper)
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)
        patch_method(torch.nn.Module, "__init__", _super_init_call)
        for mod, _orig_reset_call in _reset_call_register.items():
            setattr(mod, "reset_parameters", _orig_reset_call)

        _escaped_mods = list()
        ignore_save = (not disk_offload) or (torch.distributed.is_initialized() and local_rank() != 0)
        # check modules not successfully registered
        for root_mod in _mods_to_clean:
            for mod in root_mod.modules():
                _escaped_mods.append(mod)
                _set_model_checkpoint_name(mod)
                empty_param(mod, ignore_save=ignore_save)

        all_mods = _escaped_mods + _mods_to_clean
        # We cannot assume that the first of the _mods_to_clean is the root module
        for root_mod in all_mods:
            _tied_parameters = _find_tied_weights_for_meta(root_mod)
            _retie_weights(root_mod, _tied_parameters)
        # clear _TIE_DICT
        _TIE_DICT.clear()
        gc.collect()
        if torch.distributed.is_initialized():
            logger.info(f"Rank {rank()} wait for meta init offloading to complete.")

            # local rank 0 makes a wait file, other ranks wait it to be made
            wait_file = os.path.join(_MetaModeContext.offload_path, "meta_init_wait_file")
            if int(local_rank()) == 0:
                assert not os.path.exists(wait_file), "wait file already exists"
                open(wait_file, "a").close()
            else:
                while not os.path.exists(wait_file):
                    time.sleep(10)

            # after wait file, barrier across nodes should not take too long
            torch.distributed.barrier()

            # rm wait file
            if int(local_rank()) == 0:
                os.remove(wait_file)


def _find_tied_weights_for_meta(model):
    _name_dict = dict()
    _tied_parameters = dict()
    for name, param in model.named_parameters():
        if hasattr(param, "checkpoint_name"):
            if param.checkpoint_name in _name_dict:
                _tied_parameters[name] = _name_dict[param.checkpoint_name]
            else:
                _name_dict[param.checkpoint_name] = name
        else:
            logger.warning("params have no checkpoint_name:%s", name)

    return _tied_parameters


def _find_tied_weights(model, **kwargs):
    """This is an implementation taken from HuggingFace accelerate package.
    Tied parameters are identified by their absence in root modules' named_parameters()
    """
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        result[new_name] = full_name

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        _find_tied_weights(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return result


def _retie_weights(model, _tied_parameters):
    for src, dst in _tied_parameters.items():
        dst_split = dst.split(".")
        dst_name = dst_split.pop()
        dst_module_name = ".".join(dst_split)
        dst_module = attrgetter(dst_module_name)(model)
        src_weight = attrgetter(src)(model)
        setattr(dst_module, dst_name, src_weight)


def _reload_meta_parameter(module, tensor_name, device, value=None, ckpt_name=None):
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if "checkpoint_name" in kwargs:
                del kwargs["checkpoint_name"]
            new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            if ckpt_name is not None:
                setattr(new_value, "checkpoint_name", ckpt_name)
            module._parameters[tensor_name] = new_value


def is_meta(module):
    return any(t.is_meta for t in module.parameters()) or any(t.is_meta for t in module.buffers())


def get_checkpoint_name(prefix_name="", is_param=True):
    if is_param:
        cur_v = _MetaModeContext.param_counter
        _MetaModeContext.param_counter += 1
    else:
        cur_v = _MetaModeContext.buffer_counter
        _MetaModeContext.buffer_counter += 1
    type_name = "param" if is_param else "buffer"
    return f"{_MetaModeContext.offload_path}/{prefix_name}{type_name}_{cur_v}.bin"


def _set_model_checkpoint_name(mod, prefix_name=""):
    for name in mod._parameters:
        if mod._parameters[name] is not None:
            if not hasattr(mod._parameters[name], "checkpoint_name"):
                if mod._parameters[name] in _TIE_DICT:
                    ckpt_name = _TIE_DICT[mod._parameters[name]]
                else:
                    ckpt_name = get_checkpoint_name(prefix_name, is_param=True)
                setattr(mod._parameters[name], "checkpoint_name", ckpt_name)

    for name in mod._buffers:
        if mod._buffers[name] is not None:
            if not hasattr(mod._buffers[name], "checkpoint_name"):
                if mod._buffers[name] in _TIE_DICT:
                    ckpt_name = _TIE_DICT[mod._buffers[name]]
                else:
                    ckpt_name = get_checkpoint_name(prefix_name, is_param=False)
                setattr(mod._buffers[name], "checkpoint_name", ckpt_name)


def empty_param(mod, prefix_name="", ignore_save=False):
    for name in mod._parameters:
        if mod._parameters[name] is not None:
            param_cls = type(mod._parameters[name])
            param_kwargs = mod._parameters[name].__dict__
            if not hasattr(mod._parameters[name], "checkpoint_name"):
                if mod._parameters[name] in _TIE_DICT:
                    ckpt_name = _TIE_DICT[mod._parameters[name]]
                else:
                    ckpt_name = get_checkpoint_name(prefix_name, is_param=True)
                setattr(mod._parameters[name], "checkpoint_name", ckpt_name)
            else:
                ckpt_name = mod._parameters[name].checkpoint_name
            if (
                not torch.distributed.is_initialized()
                or (local_rank() is not None and int(local_rank()) == 0 and not ignore_save)
            ) and mod._parameters[name].device != torch.device("meta"):
                torch.save(mod._parameters[name], ckpt_name)
            delattr(mod._parameters[name], "checkpoint_name")
            mod._parameters[name] = param_cls(
                mod._parameters[name].to(torch.device("meta")),
                requires_grad=mod._parameters[name].requires_grad,
                **param_kwargs,
            )
            setattr(mod._parameters[name], "checkpoint_name", ckpt_name)
    for name in mod._buffers:
        if mod._buffers[name] is not None:
            if not hasattr(mod._buffers[name], "checkpoint_name"):
                if mod._buffers[name] in _TIE_DICT:
                    ckpt_name = _TIE_DICT[mod._buffers[name]]
                else:
                    ckpt_name = get_checkpoint_name(prefix_name, is_param=False)
                setattr(mod._buffers[name], "checkpoint_name", ckpt_name)
            else:
                ckpt_name = mod._buffers[name].checkpoint_name
            if (
                not torch.distributed.is_initialized()
                or (local_rank() is not None and int(local_rank()) == 0 and not ignore_save)
            ) and mod._buffers[name].device != torch.device("meta"):
                torch.save(mod._buffers[name], ckpt_name)
            delattr(mod._buffers[name], "checkpoint_name")
            mod._buffers[name] = mod._buffers[name].to(torch.device("meta"))
            setattr(mod._buffers[name], "checkpoint_name", ckpt_name)


def recursive_empty_param(mod, prefix_name="", ignore_save=False):
    for _, m in mod.named_modules():
        empty_param(m, prefix_name=prefix_name, ignore_save=ignore_save)


def reload_meta_module(module, device="cpu", delete_ckpt_name=True, retie_weights=True):
    """Reload a meta module to device, possibly do a retie_weights operation.

    Args:
        module (torch.nn.Module): The module to be reloaded.
        device (str, torch.device): The device onto which the reloaded parameters should be put onto.
        delete_ckpt_name (bool): Whether to delete the checkpoint_name attribute.
        retie_weights (bool): Whether to perform retie weights operation. The search of tied parameters
            is only performed over "module". If some parameters of this module is tied to parameters outside
            the scope of this module, retie_weights operation will take no effect.
    """
    if retie_weights:
        # Here use _find_tied_weights since there is no guarantee how modules are initialized
        _tied_parameters = _find_tied_weights(module)

    if not is_meta(module):
        module.to(device)
        # Moving modules with .to() method does not break the tying relation unless device is meta
        if (device == "meta" or device == torch.device("meta")) and retie_weights:
            _retie_weights(module, _tied_parameters)
        return

    def _loop_and_reload(module):
        for name, param in module.named_parameters():
            if param.device == torch.device("meta"):
                if hasattr(param, "checkpoint_name"):
                    if Path(param.checkpoint_name).exists():
                        loaded_param = torch.load(param.checkpoint_name)
                    elif _MetaModeContext._SHARD_FLAT_PARAM_LOADER:
                        if hasattr(param, "sync_module_states"):
                            loaded_param = _MetaModeContext.get_current_shard_flat_loader().load_tensor_by_name(
                                param.checkpoint_name, sync_module_states=True
                            )
                            delattr(param, "sync_module_states")
                        elif hasattr(param, "intra_load"):
                            loaded_param = _MetaModeContext.get_current_shard_flat_loader().load_with_intra_nodes(
                                param.checkpoint_name
                            )
                            delattr(param, "intra_load")
                        else:
                            loaded_param = _MetaModeContext.get_current_shard_flat_loader().load_tensor_by_name(
                                param.checkpoint_name
                            )
                    else:
                        raise ValueError(
                            (
                                "reload meta module needs set `checkpoint_name`, for init, "
                                "you needs set checkpoint_name for a file which is saved by torch. "
                                "For load ckpt, you need to set `checkpoint_name` and use "
                                "ShardTensorUtil to load flat param."
                            )
                        )
                    if loaded_param.dtype is not param.dtype:
                        loaded_param = loaded_param.to(param.dtype)
                    # ckpt_name: set to the new Parameter, None if delete ckpt name
                    # old_ckpt_name: set to the old Parameter, in case weight tying
                    ckpt_name = param.checkpoint_name if not delete_ckpt_name else None
                    old_ckpt_name = param.checkpoint_name
                    # checkpoint_name attr must be deleted as this is a custom attr and is cannot be used
                    # to initialize the new Parameter object
                    delattr(param, "checkpoint_name")
                    _reload_meta_parameter(module, name, device, value=loaded_param, ckpt_name=ckpt_name)
                    # In case param is shared by other submodules, old_ckpt_name has to be written back
                    # so the other submodules can correctly load the param
                    setattr(param, "checkpoint_name", old_ckpt_name)
                    del loaded_param, param
                else:
                    raise ValueError(f"meta model {module} is not checkpointed, for {name}")
            elif param.device != torch.device(device):
                logger.info("reload meta param %s", name)
                _reload_meta_parameter(module, name, device)
        for name, buffer in module.named_buffers():
            if buffer.device == torch.device("meta"):
                if hasattr(buffer, "checkpoint_name"):
                    if Path(buffer.checkpoint_name).exists():
                        loaded_buffer = torch.load(buffer.checkpoint_name)
                    elif _MetaModeContext._SHARD_FLAT_PARAM_LOADER:
                        loaded_buffer = _MetaModeContext.get_current_shard_flat_loader().load_tensor_by_name(
                            buffer.checkpoint_name
                        )
                    else:
                        raise ValueError(
                            (
                                "reload meta module needs set `checkpoint_name`, for init, "
                                "you needs set checkpoint_name for a file which is saved by torch. "
                                "For load ckpt, you need to set `checkpoint_name` and use "
                                "ShardTensorUtil to load flat param."
                            )
                        )
                    if loaded_buffer.dtype is not buffer.dtype:
                        loaded_buffer = loaded_buffer.to(buffer.dtype)
                    ckpt_name = buffer.checkpoint_name if not delete_ckpt_name else None
                    old_ckpt_name = buffer.checkpoint_name
                    delattr(buffer, "checkpoint_name")
                    _reload_meta_parameter(module, name, device, value=loaded_buffer, ckpt_name=ckpt_name)
                    setattr(buffer, "checkpoint_name", ckpt_name)
                    del loaded_buffer, buffer
                else:
                    raise ValueError(f"meta model is not checkpointed, for {name}")
            elif buffer.device != torch.device(device):
                logger.info("reload meta buffer %s", name)
                _reload_meta_parameter(module, name, device)

        for child in module.children():
            _loop_and_reload(child)

    _loop_and_reload(module)

    # calling to(device) in case module contains some funky attributes
    module.to(device)
    if retie_weights:
        _retie_weights(module, _tied_parameters)


def move_args_kwargs_to_device(args, kwargs, device="cpu"):
    def transform_fn(input_):
        if isinstance(input_, torch.Tensor):
            return input_.to(device)
        else:
            return input_

    args = map_aggregate(args, transform_fn)
    kwargs = map_aggregate(kwargs, transform_fn)
    return args, kwargs


def deepcopy_checkpoint_name(model_copy, model):
    orig_parameters = dict(model.named_parameters())
    orig_buffers = dict(model.named_buffers())
    for name, param in model_copy.named_parameters():
        if param.device == torch.device("meta"):
            if not hasattr(param, "checkpoint_name") and hasattr(orig_parameters[name], "checkpoint_name"):
                setattr(param, "checkpoint_name", orig_parameters[name].checkpoint_name)
    for name, buffer in model_copy.named_buffers():
        if buffer.device == torch.device("meta"):
            if not hasattr(buffer, "checkpoint_name") and hasattr(orig_buffers[name], "checkpoint_name"):
                setattr(buffer, "checkpoint_name", orig_buffers[name].checkpoint_name)


@contextmanager
def record_module_init():
    """
    Record modules' init args and kwargs while meta constructing model. Since we don't
    save or offload the initial weight, we should reset_paramters or (hf)_init_weights
    after building the real modules with the recorded args/kwargs.
    This contextmanager was originally designed for building deepspeed PipelineModule from
    native torch model implementation.
    """

    def init_record_helper(f):
        @functools.wraps(f)
        def wrapper(module: torch.nn.Module, *args, **kwargs):
            f(module, *args, **kwargs)
            # record args/kwargs after original init, in case parent cls init covers them
            # in mistake; it must be satisfied that args/kwargs not changed in init
            module._init_args = args
            module._init_kwargs = kwargs
            # torch.device('meta') contextmanager may not handle nn.Parameter(...),
            # .to('meta') manually to force everything in meta
            module.to("meta")

        return wrapper

    def _enable_class(cls):
        cls._old_init = cls.__init__
        cls.__init__ = init_record_helper(cls.__init__)

    def _disable_class(cls):
        cls.__init__ = cls._old_init
        delattr(cls, "_old_init")

    def _init_subclass(cls, **kwargs):
        cls.__init__ = init_record_helper(cls.__init__)

    def substitute_init_recursively(cls, func, visited):
        for subcls in cls.__subclasses__():
            substitute_init_recursively(subcls, func, visited)
            if subcls not in visited:
                func(subcls)
                visited.add(subcls)

    try:
        substitute_init_recursively(torch.nn.modules.module.Module, _enable_class, set())
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        # torch meta init
        torch.device("meta").__enter__()
        yield
    finally:
        substitute_init_recursively(torch.nn.modules.module.Module, _disable_class, set())
        torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
        delattr(torch.nn.modules.module.Module, "_old_init_subclass")
        torch.device("meta").__exit__()


def build_recorded_module(meta_module):
    """
    Build the real module from the recorded meta module, supports recursively building
    the meta submodules in args/kwargs.
    Support custom build function and post process function.
    """
    if len(meta_module._parameters) == 0 and len(meta_module._buffers) == 0:
        # Module without param/buffer, regards itself as builded module after build child modules
        assert not hasattr(
            meta_module, "_build_fn"
        ), f"module {meta_module.__class__.__name__} without param/buffer should have not _build_fn."
        memos = dict()
        for child_name, child_module in meta_module._modules.items():
            if child_module not in memos:
                memos[child_module] = child_name
                meta_module._modules[child_name] = build_recorded_module(child_module)
            else:
                memoried_name = memos[child_module]
                meta_module._modules[child_name] = meta_module._modules[memoried_name]
        builded_module = meta_module
    else:
        # Build from init args/kwargs, check if has children module
        if len(meta_module._modules) != 0:
            logger.info(
                f"Meta_module {meta_module.__class__.__name__} has its own param/buffer "
                f"{[k for k in chain(meta_module._parameters, meta_module._buffers)]}, "
                f"but has submodules {[k for k in meta_module._modules]}. Building it "
                f"from init args/kwargs may lead to coarse-grained materialization (OOM) "
                f"and repeatly building if submodule has custom _build_fn/_post_fn."
            )

        # recursively build module in args and kwargs
        assert hasattr(meta_module, "_init_args") and hasattr(
            meta_module, "_init_kwargs"
        ), "must construct meta module with record_module_init contextmanager"
        args = []
        for arg in meta_module._init_args:
            if isinstance(arg, torch.nn.Module):
                arg = build_recorded_module(arg)
            args.append(arg)
        kwargs = dict()
        for k, v in meta_module._init_kwargs.items():
            if isinstance(v, torch.nn.Module):
                v = build_recorded_module(v)
            kwargs[k] = v

        # support custom build fn
        if hasattr(meta_module, "_build_fn"):
            build_callable = meta_module._build_fn
        else:
            build_callable = meta_module.__class__
        builded_module = build_callable(*args, **kwargs)

        # if submodules have custom _build_fn/_post_fn, rebuild and substitute them
        for submodule_name, submodule in list(meta_module.named_modules())[1:]:
            if hasattr(submodule, "_build_fn") or hasattr(submodule, "_post_fn"):
                builded_submodule = build_recorded_module(submodule)
                recursive_setattr(builded_module, submodule_name, builded_submodule)

    # support custom post process fn
    if hasattr(meta_module, "_post_fn"):
        builded_module = meta_module._post_fn(builded_module)

    return builded_module
