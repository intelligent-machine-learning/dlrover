from enum import auto
from typing import Callable, Dict, List


class ATorchHooks(object):
    COMPUTE_GPU_UTIL_HOOK = auto()
    REPORT_METRICS_HOOK = auto()
    ADDITIONAL_TENSORBOARD_HOOK = auto()

    # hooks stored as dict, key for hook_type, value for corresponding hook list.
    hooks: Dict[auto, List[Callable]] = {}

    @staticmethod
    def register_hook(hook_type, hook_func):
        if hook_type not in ATorchHooks.hooks:
            ATorchHooks.hooks[hook_type] = [hook_func]
        elif hook_func not in ATorchHooks.hooks[hook_type]:
            ATorchHooks.hooks[hook_type].append(hook_func)

    @staticmethod
    def remove_hook(hook_type, hook_func):
        if hook_type in ATorchHooks.hooks and hook_func in ATorchHooks.hooks[hook_type]:
            ATorchHooks.hooks[hook_type].remove(hook_func)

    @staticmethod
    def call_hooks(hook_type, *args, **kargs):
        if hook_type in ATorchHooks.hooks:
            for func in ATorchHooks.hooks[hook_type]:
                func(*args, **kargs)
