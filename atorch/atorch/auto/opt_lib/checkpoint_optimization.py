from functools import partial
from typing import Callable

import torch

from atorch.auto.opt_lib.optimization import Optimization
from atorch.auto.opt_lib.utils import to_module_class_by_name
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import parallel_group
from atorch.modules.distributed_modules.activation_checkpointing import tp_wrap_fn
from atorch.utils.version import torch_version


class CheckpointOptimization(Optimization):
    """CheckpointOptimization will checkpoint modules who are instance of config.
    config is a module class or a tuple of module classes.
    """

    checkpoint_funcs_before_overrided = None

    def __init__(self):
        super().__init__("checkpoint", "checkpoint", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config, apply_wrapper=False)
        return True, config, model_context

    def transform(self, model_context, config=None):
        if torch_version() < (1, 12, 1):
            logger.warning(
                f"Checkpoint requires torch version >= 1.12.1, ignored as current version is {torch_version()}."
            )
        else:
            model_context.add_wrapper(
                "checkpoint", CheckpointOptimization.apply_wrapper, wrapper_config=config, is_pre_wrapper=False
            )
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        # wrapper_config should be a module class or tuple of module classes
        if torch_version() == (1, 12, 1):
            # 1.12.1 does not have apply_activation_checkpointing, and checkpoint does not support kwargs.
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
            from torch.utils.checkpoint import CheckpointFunction, _checkpoint_without_reentrant

            # Support kwargs in checkpoint
            def atorch_checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
                # Hack to mix *args with **kwargs in a python 2.7-compliant way
                preserve = kwargs.pop("preserve_rng_state", True)

                # Hack to put kwargs into args
                all_args = list(args)
                if kwargs:
                    for x in kwargs:
                        all_args.append(kwargs[x])

                if use_reentrant:
                    return CheckpointFunction.apply(function, preserve, *all_args)
                else:
                    return _checkpoint_without_reentrant(function, preserve, *all_args)

            # Override hack
            CheckpointOptimization.checkpoint_funcs_before_overrided = (
                torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint
            )
            torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint = atorch_checkpoint

            def lambda_auto_wrap_policy(
                module: torch.nn.Module, recurse: bool, unwrapped_params: int, lambda_fn: Callable
            ) -> bool:
                if recurse:
                    return True  # always recurse
                return lambda_fn(module)

            # Provide apply_activation_checkpointing
            def apply_activation_checkpointing(model, checkpoint_wrapper_fn=CheckpointWrapper, check_fn=lambda _: True):
                from torch.distributed.fsdp.wrap import _recursive_wrap

                _recursive_wrap(
                    module=model,
                    auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
                    wrapper_cls=checkpoint_wrapper_fn,
                    ignored_modules=set(),
                    ignored_params=set(),
                    only_wrap_children=True,
                )

        else:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
            except ImportError:
                logger.warning("Checkpoint not supported, thus ignored!")
                return model_context
        # wrap model

        if not isinstance(wrapper_config, tuple):
            wrapper_config = (wrapper_config,)
        wrapper_config = to_module_class_by_name(model_context.model, wrapper_config)

        def check_fn(module):
            return isinstance(module, wrapper_config)

        # If tensor parallel group exists, use tensor parallel checkpoint method
        tp_group = parallel_group("tensor")
        if tp_group is None:
            apply_activation_checkpointing(model_context.model, check_fn=check_fn)
        else:
            apply_activation_checkpointing(
                model_context.model,
                checkpoint_wrapper_fn=tp_wrap_fn,
                check_fn=check_fn,
            )

        return model_context

    @staticmethod
    def reset(config):
        if CheckpointOptimization.checkpoint_funcs_before_overrided is None:
            return
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint = (
            CheckpointOptimization.checkpoint_funcs_before_overrided
        )
        CheckpointOptimization.checkpoint_funcs_before_overrided = None
