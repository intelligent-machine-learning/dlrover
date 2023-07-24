import collections
import sys
from functools import partialmethod, wraps

import torch
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from atorch.distributed.distributed import parallel_group
from atorch.utils.version import torch_version

if torch_version() >= (1, 12, 0):
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

else:
    from fairscale.optim.grad_scaler import ShardedGradScaler


try:
    from apex.amp import _amp_state

    from atorch.amp import initialize, scale_loss

    _amp_apex_is_available = True
except ImportError:
    _amp_apex_is_available = False
    _amp_state = None
from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import local_rank


class AmpApexOptimization(Optimization):
    activated = False

    def __init__(self, name):
        super().__init__(name, "amp", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config, apply_wrapper=False)
        return True, config, model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        if wrapper_name.startswith("amp_apex_"):
            device = f"cuda:{local_rank()}"
            torch.cuda.set_device(device)
            model = model_context.model.to(device)
            optimizer = model_context.optim
            model, optimizer = initialize(model, optimizers=optimizer, **wrapper_config)
            AutoAccelerateContext.add_ac_attr(f"{wrapper_name}_optim", optimizer)
            original_loss_func = model_context.loss_func
            model_context.model = model
            model_context.optim = optimizer
            original_loss_func = model_context.loss_func
            if original_loss_func is not None:
                model_context.loss_func = amp_apex_loss_func(original_loss_func)

    @staticmethod
    def reset():
        if AmpApexOptimization.activated is False:
            return
        # reset apex _amp_state
        if _amp_state is not None and hasattr(_amp_state, "opt_properties"):
            if _amp_state.opt_properties.enabled is True:
                _amp_state.opt_properties.enabled = False
            if _amp_state.opt_properties.opt_level is not None:
                _amp_state.opt_properties.opt_level = None
            if _amp_state.opt_properties.cast_model_type is not None:
                _amp_state.opt_properties.cast_model_type is None
            if _amp_state.opt_properties.patch_torch_functions is True:
                _amp_state.opt_properties.patch_torch_functions = False
            if _amp_state.opt_properties.keep_batchnorm_fp32 is not None:
                _amp_state.opt_properties.keep_batchnorm_fp32 = None
            if _amp_state.opt_properties.master_weights is not None:
                _amp_state.opt_properties.master_weights = None
            if _amp_state.opt_properties.loss_scale is not None:
                _amp_state.opt_properties.loss_scale = 1.0
            if _amp_state.min_loss_scale is not None:
                _amp_state.min_loss_scale = None
            if _amp_state.max_loss_scale is not None:
                _amp_state.max_loss_scale = 2.0**24
            # reset funcs overrided by apex
            _amp_state.handle._deactivate()
        AmpApexOptimization.activated = False

    @staticmethod
    def activate_apex_amp():
        if AmpApexOptimization.activated is True:
            return
        AmpApexOptimization.activated = True


class AmpApexO1Optimization(AmpApexOptimization):
    def __init__(self):
        super().__init__("amp_apex_o1")

    def transform(self, model_context, config=None, apply_wrapper=False):
        if not _amp_apex_is_available or not torch.cuda.is_available():
            return model_context
        if (
            _amp_state is not None
            and hasattr(_amp_state, "opt_properties")
            and _amp_state.opt_properties.enabled is True
        ):
            raise RuntimeError("Cannot use amp_apex optimization multiple times.")
        self.activate_apex_amp()
        config = {"opt_level": "O1", "loss_scale": "dynamic"}
        model_context.add_wrapper("amp_apex_o1", AmpApexO1Optimization.apply_wrapper, config, is_pre_wrapper=False)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        super(AmpApexO1Optimization, AmpApexO1Optimization).apply_wrapper(model_context, wrapper_name, wrapper_config)

    @staticmethod
    def reset(_):
        super(AmpApexO1Optimization, AmpApexO1Optimization).reset()


class AmpApexO2Optimization(AmpApexOptimization):
    def __init__(self):
        super().__init__("amp_apex_o2")

    def transform(self, model_context, config=None, apply_wrapper=False):
        if not _amp_apex_is_available or not torch.cuda.is_available():
            return model_context
        if _amp_state is not None and hasattr(_amp_state, "opt_properties") and _amp_state.opt_properties.enabled:
            raise RuntimeError("Cannot use amp_apex optimization multiple times.")
        self.activate_apex_amp()
        config = {"opt_level": "O2", "loss_scale": "dynamic"}
        model_context.add_wrapper("amp_apex_o2", AmpApexO2Optimization.apply_wrapper, config, is_pre_wrapper=False)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        super(AmpApexO2Optimization, AmpApexO2Optimization).apply_wrapper(model_context, wrapper_name, wrapper_config)

    @staticmethod
    def reset(_):
        super(AmpApexO2Optimization, AmpApexO2Optimization).reset()


class ApexAmpScaledLoss(torch.Tensor):
    """
    Inherit torch.Tensor for amp scaled loss.

    To use atorch.amp, users have to change their codes as following:

        # change 1: wrap model, optimizer
        model, optimizer = amp.initialize(model, optimizer, ...)
        for data in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(data[1], output)

            # change 2: use `with` statement
            with amp.scale_loss(loss, optimizer) as sl:
                sl.backward()

            optimizer.step()

    We cannot force users use `with amp.scale_loss` statement, so we override the `loss` Tensor's backward method. By
    combining `AmpScaledLoss` with `amp_loss_func`, we are able to let users call `loss.backward()` as usual.
    """

    def backward(self):
        if hasattr(AutoAccelerateContext, "amp_apex_o1_optim"):
            amp_apex_optim = AutoAccelerateContext.amp_apex_o1_optim
        elif hasattr(AutoAccelerateContext, "amp_apex_o2_optim"):
            amp_apex_optim = AutoAccelerateContext.amp_apex_o2_optim
        with scale_loss(self, amp_apex_optim) as scaled_loss:
            self = scaled_loss
            super().backward()


def amp_apex_loss_func(loss_func):
    @wraps(loss_func)
    def amp_loss_wrapper(*args, **kwargs):
        loss_result = loss_func(*args, **kwargs)
        if isinstance(loss_result, collections.abc.Sequence):
            loss = loss_result[0]
        else:
            loss = loss_result
        loss = loss.as_subclass(ApexAmpScaledLoss)
        if isinstance(loss_result, collections.abc.Sequence):
            return loss, *loss_result[1:]
        else:
            return loss

    return amp_loss_wrapper


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
            if parallel_group_zero is not None and parallel_group_data is not None:
                grad_scaler = ShardedGradScaler(process_group=parallel_group_zero)
            elif parallel_group_data is not None and isinstance(model, (ShardedDDP, FSDP)):
                grad_scaler = ShardedGradScaler(process_group=parallel_group_data)
            else:
                grad_scaler = GradScaler()
            if wrapper_config["dtype"] == torch.bfloat16 and skip_if_nonfinite:
                # Bf16 does not need loss scaling. We only need GradScaler's inf checking.
                grad_scaler._init_scale = 1.0
                grad_scaler._growth_factor = 1.0
                grad_scaler._backoff_factor = 1.0
                grad_scaler._growth_interval = sys.maxsize
            if optimizer is not None and loss_func is not None:
                if not hasattr(AutoAccelerateContext, "amp_native_grad_scaler"):
                    AutoAccelerateContext.add_ac_attr(
                        "amp_native_grad_scaler", {AutoAccelerateContext.counter: grad_scaler}
                    )
                else:
                    AutoAccelerateContext.amp_native_grad_scaler.update({AutoAccelerateContext.counter: grad_scaler})
                model_context.optim = AmpNativeOptimizer(optimizer, grad_scaler)
                wrapper_config["counter"] = AutoAccelerateContext.counter
                model_context.loss_func = amp_native_loss_func(loss_func, **wrapper_config)


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
        scale_before = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizer, *args, **kwargs)
        self.grad_scaler.update()
        scale_after = self.grad_scaler.get_scale()
        # If we reduced the loss scale, it means the optimizer step was skipped because of gradient overflow.
        self._is_overflow = scale_after < scale_before

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
