# coding=utf-8
from __future__ import absolute_import, unicode_literals

import functools
from typing import Union

import torch
import torch.nn as nn

from atorch.data_parallel.adp import AllDataParallel
from atorch.data_parallel.auto_wrap import auto_wrap, enable_wrap

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def wrapper_helper(
    module, wrapper_modules, min_num_params=1e8, exclude_wrap_modules=None, force_leaf_modules=None, **kwargs
):
    """
    params:
        module： nn.Module need to wrapper
        wrapper_modules: module class which will be force wrapper
        min_num_params: default:1e8.if one module.numel()>min_num_params ,
                        it will be wrapped, otherwise,it will not wrapped ant not wrap at sub modules;
        exclude_wrap_modules: exclude wrap module, but it will recursively judge sub modules
        force_leaf_modules:exclude wrap module, and it will **not** recursively judge sub modules
    return:
        AllDataParallel instance. wrapped module

    example:
        wrapper BertLayer inside self.model:
          ```self.model = wrapper_helper(self.model, BerLayer, 7e6,
          {nn.ModuleList, nn.ModuleDict, nn.Embedding},
          {nn.MultiheadAttention, nn.Linear, ResNet}),
          ```
    """

    def auto_wrap_policy(
        module: nn.Module,
        recurse: bool,
        unwrapped_params: int,
        # These are customizable for this default policy function.
        min_num_params: int = int(1e8),
        force_leaf_modules=None,
        exclude_wrap_modules=None,
    ) -> Union[bool, str]:

        is_large = unwrapped_params >= min_num_params
        if force_leaf_modules is None:
            force_leaf_modules = set()
        if exclude_wrap_modules is None:
            exclude_wrap_modules = {nn.ModuleList, nn.ModuleDict}

        if recurse:
            # We should recurse if the module is big enough but not in force_leaf_modules list.
            if isinstance(module, wrapper_modules):
                # force layer use adp,avoid checkpoint activation issue
                return "wrap_skip_child"
            is_wrapper = is_large and not isinstance(module, tuple(force_leaf_modules))
            return is_wrapper
        else:
            if isinstance(module, wrapper_modules):
                return True
            is_wrapper = is_large and not isinstance(module, tuple(exclude_wrap_modules))
            return is_wrapper

    policy = functools.partial(
        auto_wrap_policy,
        min_num_params=min_num_params,
        force_leaf_modules=force_leaf_modules,
        exclude_wrap_modules=exclude_wrap_modules,
    )

    mixed_precision = kwargs.pop("mixed_precision", False)
    flatten_parameters = kwargs.pop("flatten_parameters", True)
    reshard_after_forward = kwargs.pop("reshard_after_forward", True)
    kwargs.update(
        mixed_precision=mixed_precision,
        flatten_parameters=flatten_parameters,
        reshard_after_forward=reshard_after_forward,
    )
    with (enable_wrap(policy, wrapper_cls=AllDataParallel, **kwargs)):
        module = auto_wrap(module)
    if not isinstance(module, AllDataParallel) and any(not hasattr(p, "_is_sharded") for p in module.parameters()):
        kwargs["reshard_after_forward"] = False
        module = AllDataParallel(module, **kwargs)
    # add name to debug
    for name, m in module.named_modules():
        m.origin_name = name
    return module
