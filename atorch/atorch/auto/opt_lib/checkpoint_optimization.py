from collections.abc import MutableMapping
from functools import partial
from typing import Callable

import torch

from atorch.auto.opt_lib.optimization import Optimization
from atorch.auto.opt_lib.utils import to_module_class_by_name
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import parallel_group
from atorch.modules.distributed_modules.activation_checkpointing import tp_wrap_fn
from atorch.utils.version import package_version_smaller_than, torch_version


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
        # A HACK to read in amp_config for tp checkpointing
        from atorch.auto.auto_accelerate_context import AutoAccelerateContext

        max_checkpoint_module_num = None
        if isinstance(wrapper_config, MutableMapping):
            temp = wrapper_config
            wrapper_config = wrapper_config["wrap_class"]
            other_config = temp
            if "max_checkpoint_module_num" in other_config:
                max_checkpoint_module_num = other_config["max_checkpoint_module_num"]
        else:
            # this is for compatiable
            other_config = {}

        counter = AutoAccelerateContext.counter
        if hasattr(AutoAccelerateContext, "tp_amp_config"):
            amp_config = AutoAccelerateContext.tp_amp_config.get(counter, None)
        else:
            amp_config = None
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
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                fp8_enabled = hasattr(AutoAccelerateContext, "fp8_enabled") and AutoAccelerateContext.fp8_enabled.get(
                    AutoAccelerateContext.counter, False
                )

                if fp8_enabled:
                    # use te checkpoint
                    from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
                    from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

                    # patch te checkpoint to support non-tensor inputs/outputs if needed
                    from atorch.utils.patch_te import patch_te_if_needed

                    patch_te_if_needed()

                    _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()

                    no_reentrant = other_config.pop("no_reentrant", False)

                    def get_cuda_rng_tracker():
                        return _CUDA_RNG_STATE_TRACKER

                    def te_checkpoint_func(m, *args, **kargs):
                        # te 1.5+ has different checkpoint interface.
                        if package_version_smaller_than("transformer_engine", "1.5"):
                            return te_checkpoint(m, False, get_cuda_rng_tracker, None, *args, **kargs)
                        return te_checkpoint(
                            m,
                            *args,
                            distribute_saved_activations=False,
                            get_rng_state_tracker=get_cuda_rng_tracker,
                            tp_group=None,
                            use_reentrant=not no_reentrant,
                            **kargs,
                        )

                    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_fn=te_checkpoint_func)
                    apply_activation_checkpointing = partial(
                        apply_activation_checkpointing, checkpoint_wrapper_fn=checkpoint_wrapper_fn
                    )
                else:
                    assert other_config is not None  # for mypy
                    # pytorch uses default None for use_reentrant parameter in checkpoint,
                    # which is equivalent to reentrant. This parameter will become mandatory in torch 2.4.
                    no_reentrant = other_config.pop("no_reentrant", None)
                    selective_offload = other_config.pop("selective_offload", None)
                    if no_reentrant is None:
                        # Used no_reentrant for checkpoint selective offload
                        no_reentrant = selective_offload is not None
                        logger.warning(
                            "checkpoint config does not contains no_reentrant value, "
                            "set no_reentrant=True as selective_offload is used"
                            if selective_offload is not None
                            else "set no_reentrant=False as default"
                        )
                    else:
                        logger.info(f"checkpoint config contains no_reentrant={no_reentrant}")
                    checkpoint_impl = CheckpointImpl.NO_REENTRANT if no_reentrant else CheckpointImpl.REENTRANT
                    checkpoint_wrapper_fn_kwargs = {"checkpoint_impl": checkpoint_impl}
                    if selective_offload is not None:
                        if checkpoint_impl == CheckpointImpl.REENTRANT:
                            raise ValueError(
                                "selective offloading don't support `CheckpointImpl.REENTRANT`, "
                                "requires no_reentrant=False in checkpoint config"
                            )
                        if "offload_args" not in selective_offload or "num_layers" not in selective_offload:
                            raise ValueError("`offload_args` or `num_layers` is not passed")

                        from .selective_offloading_checkpoint import (
                            OffloadOpManagerArgs,
                            get_selective_offloading_checkpoint_modes,
                        )

                        args = [OffloadOpManagerArgs(*arg) for arg in selective_offload["offload_args"]]
                        num_layers = selective_offload["num_layers"]
                        context_fn = get_selective_offloading_checkpoint_modes(args, num_layers)
                        checkpoint_wrapper_fn_kwargs["context_fn"] = context_fn
                        logger.info(f"selective_offloading_checkpoint is on, {selective_offload}")

                    checkpoint_wrapper_fn = partial(checkpoint_wrapper, **checkpoint_wrapper_fn_kwargs)
                    apply_activation_checkpointing = partial(
                        apply_activation_checkpointing, checkpoint_wrapper_fn=checkpoint_wrapper_fn
                    )
            except ImportError:
                logger.warning("Checkpoint not supported, thus ignored!")
                return model_context
        # wrap model

        if not isinstance(wrapper_config, tuple):
            wrapper_config = (wrapper_config,)
        wrapper_config = to_module_class_by_name(model_context.model, wrapper_config)

        class _check_fn:
            def __init__(self, _max_checkpoint_module_num, _wrapper_config):
                self.max_checkpoint_module_num = _max_checkpoint_module_num
                self.wrapper_config = _wrapper_config
                self.checkpointed_module_num = 0
                self.skiped_module_num = 0

            def check_fn(self, module):
                if isinstance(module, self.wrapper_config):
                    if (
                        self.max_checkpoint_module_num is None
                        or self.checkpointed_module_num < self.max_checkpoint_module_num
                    ):
                        self.checkpointed_module_num += 1
                        return True
                    else:
                        self.skiped_module_num += 1
                        return False

                return False

        check_fn_instance = _check_fn(max_checkpoint_module_num, wrapper_config)
        check_fn = check_fn_instance.check_fn

        # If tensor parallel group exists, use tensor parallel checkpoint method
        tp_group = parallel_group("tensor")
        if tp_group is None:
            apply_activation_checkpointing(model_context.model, check_fn=check_fn)
        else:
            amp_config.pop("skip_if_nonfinite", None)
            apply_activation_checkpointing(
                model_context.model,
                checkpoint_wrapper_fn=partial(tp_wrap_fn, amp_config=amp_config),
                check_fn=check_fn,
            )

        if max_checkpoint_module_num is not None:
            logger.info(
                "max_checkpoint_module_num={}, checkpointed_num={}, skipped_num={}".format(
                    max_checkpoint_module_num,
                    check_fn_instance.checkpointed_module_num,
                    check_fn_instance.skiped_module_num,
                )
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
