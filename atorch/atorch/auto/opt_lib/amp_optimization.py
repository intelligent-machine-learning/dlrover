import collections
import copy
import types
from functools import partialmethod, wraps

import torch
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import atorch
from atorch.distributed.distributed import parallel_group
from atorch.utils.grad_scaler import BF16GradScaler, BF16ShardedGradScaler
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.version import torch_version

if torch_version() >= (1, 12, 0):  # type: ignore
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

else:
    from fairscale.optim.grad_scaler import ShardedGradScaler

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger


class AmpNativeOptimization(Optimization):
    def __init__(self):
        super().__init__("amp_native", "amp", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        if config is not None:
            half_dtype = config.get("dtype", None)
            if half_dtype in ("bf16", torch.bfloat16):
                if not torch.cuda.is_bf16_supported():
                    gpu_model = torch.cuda.get_device_properties(torch.cuda.current_device()).name
                    logger.info(f"Current device {gpu_model} does not support bfloat16. Set dtype to float16.")
                    config["dtype"] = torch.float16
                else:
                    config["dtype"] = torch.bfloat16
                    config.setdefault("skip_if_nonfinite", False)
            elif half_dtype in ("fp16", torch.float16, None):
                config["dtype"] = torch.float16
            else:
                raise ValueError(
                    "Invalid dtype. Supported dtypes are 'fp16', 'bf16', torch.float16, torch.bfloat16 but got"
                    f" {half_dtype}"
                )
        else:
            config = {"enabled": True, "dtype": torch.float16}
        model_context.add_wrapper("amp_native", AmpNativeOptimization.apply_wrapper, config, is_pre_wrapper=False)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        if wrapper_name == "amp_native":
            model = model_context.model
            skip_if_nonfinite = (
                wrapper_config.pop("skip_if_nonfinite") if "skip_if_nonfinite" in wrapper_config else False
            )
            if is_torch_npu_available() and skip_if_nonfinite:
                skip_if_nonfinite = False
                logger.info("NPU does not support 'skip_if_nonfinite'")
            model.forward = autocast(**wrapper_config)(model.forward)
            if hasattr(model, "generate") and callable(getattr(model, "generate")):
                model.generate = autocast(**wrapper_config)(model.generate)
            loss_func = model_context.loss_func

            if wrapper_config["dtype"] == torch.bfloat16 and not skip_if_nonfinite:
                # bfloat16 does not need loss or gradient scaling
                if loss_func is not None:
                    model_context.loss_func = autocast(**wrapper_config)(loss_func)
                return
            optimizer = model_context.optim
            parallel_group_data = parallel_group("data")
            parallel_group_zero = parallel_group("zero")

            if wrapper_config["dtype"] == torch.bfloat16 and skip_if_nonfinite:
                # Bf16 does not need loss scaling. We only need GradScaler's inf checking.
                if parallel_group_zero is not None and parallel_group_data is not None:
                    grad_scaler = BF16ShardedGradScaler(process_group=parallel_group_zero)
                elif parallel_group_data is not None and isinstance(model, (ShardedDDP, FSDP)):
                    grad_scaler = BF16ShardedGradScaler(process_group=parallel_group_data)
                else:
                    grad_scaler = BF16GradScaler()
            else:
                if parallel_group_zero is not None and parallel_group_data is not None:
                    grad_scaler = ShardedGradScaler(process_group=parallel_group_zero)
                elif parallel_group_data is not None and isinstance(model, (ShardedDDP, FSDP)):
                    grad_scaler = ShardedGradScaler(process_group=parallel_group_data)
                else:
                    grad_scaler = GradScaler()
            if optimizer is not None and loss_func is not None:
                if not hasattr(AutoAccelerateContext, "amp_native_grad_scaler"):
                    AutoAccelerateContext.add_ac_attr(
                        "amp_native_grad_scaler", {AutoAccelerateContext.counter: grad_scaler}
                    )
                else:
                    AutoAccelerateContext.amp_native_grad_scaler.update({AutoAccelerateContext.counter: grad_scaler})
                model_context.optim = AmpNativeOptimizer(optimizer, grad_scaler)
                config_copy = copy.copy(wrapper_config)
                config_copy["counter"] = AutoAccelerateContext.counter
                model_context.loss_func = amp_native_loss_func(loss_func, **config_copy)


class AmpNativaScaledLoss(torch.Tensor):
    pass


def hook_amp_native_loss_backward(self, counter):
    scaler = AutoAccelerateContext.amp_native_grad_scaler[counter]
    self = scaler.scale(self)
    super(self.__class__, self).backward()


def amp_native_loss_func(loss_func, **amp_config):
    counter = amp_config.pop("counter")

    @wraps(loss_func)
    def amp_native_loss_wrapper(*args, **kwargs):
        casted_loss_func = autocast(**amp_config)(loss_func)
        loss_result = casted_loss_func(*args, **kwargs)
        if isinstance(loss_result, collections.abc.Sequence):
            loss = loss_result[0]
        else:
            loss = loss_result
        partial_backward_func = partialmethod(hook_amp_native_loss_backward, counter=counter)
        setattr(AmpNativaScaledLoss, "backward", partial_backward_func)
        scaled_loss = loss.as_subclass(AmpNativaScaledLoss)
        if isinstance(loss_result, collections.abc.Sequence):
            return scaled_loss, *loss_result[1:]
        else:
            return scaled_loss

    return amp_native_loss_wrapper


class AmpNativeOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, grad_scaler):
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self._is_overflow = False

    def step(self, *args, **kwargs):
        self._is_overflow = False
        scale_before = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizer, *args, **kwargs)
        self.grad_scaler.update()
        scale_after = self.grad_scaler.get_scale()
        # If we reduced the loss scale, it means the optimizer step was skipped because of gradient overflow.
        self._is_overflow = scale_after < scale_before
        if hasattr(self.grad_scaler, "has_overflow"):
            self._is_overflow = self.grad_scaler.has_overflow()

    def unscale_(self):
        self.grad_scaler.unscale_(self.optimizer)

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was skipped."""
        return self._is_overflow

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)

    def __setstate__(self, state):
        return self.optimizer.__setstate__(state)


# RNG tracker object with lazy init.
_CUDA_RNG_STATE_TRACKER = None


def get_cuda_rng_tracker():
    global _CUDA_RNG_STATE_TRACKER
    if _CUDA_RNG_STATE_TRACKER is None:
        from transformer_engine.pytorch.distributed import CudaRNGStatesTracker

        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    return _CUDA_RNG_STATE_TRACKER


def is_fp8_available():
    if not torch.cuda.is_available():
        return False

    device_capability = torch.cuda.get_device_capability()
    return Fp8Optimization.device_supported(device_capability=device_capability)


class Fp8Optimization(Optimization):
    def __init__(self):
        super().__init__("fp8", "amp", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        if not is_fp8_available():
            logger.warning("No fp8-capable devices available, fp8 optimization is ignored!")
            return model_context

        model_context.add_wrapper("fp8", Fp8Optimization.apply_wrapper, config, is_pre_wrapper=True)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        """config:
        include: List[str], default None.
            If None, all nn.Linear module would use te.
            If not None, nn.Linear module's name should have at least one substring equals to items in include.
        exclude: List[str], default None.
            If None, all modules that passing include test would use te.
            If not None, if a nn.Linear module's name has at least one substring matches exclude, it will not use te.
        recipe.DelayedScaling's parameter:
            margin: default 0
            interval: default 1
            fp8_format: "HYBRID" (default) or "E4M3"
            amax_history_len: default 1024
            amax_compute_algo: “max” (default) or “most_recent”
            reduce_amax: default True
            override_linear_precision (te 1.5+), default (False, False, False)
            fp8_dpa (te 1.6+), default False
            fp8_mha (te 1.6+), default False
        """
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        from transformer_engine.pytorch.utils import check_dim_for_fp8_exec

        config = {} if wrapper_config is None else wrapper_config
        delayed_scaling_config = {}
        include = config.get("include", None)
        exclude = config.get("exclude", None)
        delayed_scaling_config["margin"] = config.get("margin", 0)
        delayed_scaling_config["interval"] = config.get("interval", 1)
        fp8_format = config.get("fp8_format", "HYBRID").upper()
        if fp8_format == "HYBRID":
            fp8_format_type = recipe.Format.HYBRID
        elif fp8_format == "E4M3":
            fp8_format_type = recipe.Format.E4M3
        else:
            raise f"fp8_format only supports HYBRID and E4M3, not {fp8_format}"
        delayed_scaling_config["fp8_format"] = fp8_format_type
        delayed_scaling_config["amax_history_len"] = config.get("amax_history_len", 1024)
        delayed_scaling_config["amax_compute_algo"] = config.get("amax_compute_algo", "max")
        if hasattr(recipe.DelayedScaling, "override_linear_precision"):
            delayed_scaling_config["override_linear_precision"] = config.get(
                "override_linear_precision", (False, False, False)
            )
        delayed_scaling_config["reduce_amax"] = config.get("reduce_amax", True)
        if hasattr(recipe.DelayedScaling, "fp8_dpa"):
            delayed_scaling_config["fp8_dpa"] = config.get("fp8_dpa", False)
        if hasattr(recipe.DelayedScaling, "fp8_mha"):
            delayed_scaling_config["fp8_mha"] = config.get("fp8_mha", False)

        verbose = config.get("verbose", False)

        def check_if_replace(key, module, m_include, m_exclude):
            # return (if_replace, if_excluded, if_shape_incompatible)
            if not isinstance(module, torch.nn.Linear):
                return False, False, False
            pass_check = False
            if m_exclude is not None:
                for name in m_exclude:
                    if name in key:
                        return False, True, False
            if m_include is not None:
                for name in m_include:
                    if name in key:
                        pass_check = True
                        break
                if not pass_check:
                    return False, False, False
            # Backward computation would use transposed weight, so also check transposed weight shape.
            transposed_shape = torch.Size(
                [module.weight.shape[1], module.weight.shape[0]] + list(module.weight.shape[2:])
            )
            backward_weight = torch.empty(transposed_shape, device="meta")
            if check_dim_for_fp8_exec(module.weight) and check_dim_for_fp8_exec(backward_weight):
                return True, False, False
            else:
                return False, False, True

        def get_te_module(module):
            config = {}
            new_module = None
            if isinstance(module, torch.nn.Linear):
                config["in_features"] = module.in_features
                config["out_features"] = module.out_features
                config["bias"] = hasattr(module, "bias") and module.bias is not None
                config["params_dtype"] = module.weight.dtype
                # te.Linear checks meta device by "meta" str, not torch.device. Thus, use str if meta.
                if isinstance(module.weight.device, torch.device) and module.weight.device.type == "meta":
                    config["device"] = "meta"
                else:
                    config["device"] = module.weight.device
                new_module = te.Linear(**config)
                with torch.no_grad():
                    new_module.weight.copy_(module.weight)
                    if hasattr(module.weight, "checkpoint_name"):
                        value = getattr(module.weight, "checkpoint_name")
                        setattr(new_module.weight, "checkpoint_name", value)
                    if config["bias"]:
                        new_module.bias.copy_(module.bias)
                        if hasattr(module.bias, "checkpoint_name"):
                            value = getattr(module.bias, "checkpoint_name")
                            setattr(new_module.bias, "checkpoint_name", value)
            return new_module

        def replace_module_with_te(model, m_include, m_exclude):
            replaced_num = 0
            exclude_num = 0
            shape_incompatible_num = 0
            for key, module in model.named_modules():
                if_replace, if_exclude, if_shape_incompatible = check_if_replace(key, module, m_include, m_exclude)
                if if_replace:
                    new_module = get_te_module(module)
                    parent = model.get_submodule(".".join(key.split(".")[:-1]))
                    target_name = key.split(".")[-1]
                    setattr(parent, target_name, new_module)
                    replaced_num += 1
                    if verbose and atorch.local_rank() in (0, None):
                        logger.info(f"Replacing {key} with te.Linear.")
                if if_exclude:
                    exclude_num += 1
                if if_shape_incompatible:
                    shape_incompatible_num += 1
            return replaced_num, exclude_num, shape_incompatible_num

        # step 1: replace linear with te.Linear
        te_module_num, te_exclude_num, te_shape_incompatible_num = replace_module_with_te(
            model_context.model, include, exclude
        )
        if te_module_num == 0:
            logger.warning(
                "No modules are using fp8, {} excluded, {} shape incompatible".format(
                    te_exclude_num, te_shape_incompatible_num
                )
            )
            return model_context
        else:
            logger.info(
                "[fp8] {} modules replaced by te.Linear, {} excluded, {} shape incompatible".format(
                    te_module_num, te_exclude_num, te_shape_incompatible_num
                )
            )

        # step 2: fp8_autocast forward, loss_func
        fp8_recipe = recipe.DelayedScaling(**delayed_scaling_config)

        ori_forward_func = model_context.model.forward

        def fp8_run_method(self, *args, **kargs):
            # TODO: set fp8_group properly if reduce_amax is True and tp is used
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                return ori_forward_func(*args, **kargs)

        model_context.model.forward = types.MethodType(fp8_run_method, model_context.model)

        if model_context.loss_func is not None:
            old_loss_func = model_context.loss_func

            def fp8_run(*args, **kargs):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    return old_loss_func(*args, **kargs)

            model_context.loss_func = fp8_run

        # step 3: record fp8_enabled for checkpoint optimization
        if hasattr(AutoAccelerateContext, "fp8_enabled"):
            AutoAccelerateContext.fp8_enabled.update({AutoAccelerateContext.counter: True})
        else:
            AutoAccelerateContext.add_ac_attr("fp8_enabled", {AutoAccelerateContext.counter: True})

        return model_context

    @staticmethod
    def device_supported(config=None, device_capability=None):
        # GPU sm version >= 8.9 (Ada, Hopper, etc)
        return isinstance(device_capability, tuple) and device_capability >= (8, 9)
