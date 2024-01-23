import collections
import copy
from functools import partialmethod, wraps

import torch
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from atorch.distributed.distributed import parallel_group
from atorch.utils.grad_scaler import BF16GradScaler, BF16ShardedGradScaler
from atorch.utils.version import torch_version

if torch_version() >= (1, 12, 0):
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
