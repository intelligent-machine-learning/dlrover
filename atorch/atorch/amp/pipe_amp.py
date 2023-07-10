import collections
import inspect
import threading
import warnings
from contextlib import contextmanager

import torch
import torch.distributed as dist

try:
    from pippy.PipelineDriver import PipeStageExecutor
except ImportError:
    PipeStageExecutor = None

from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import OptState

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.amp_optimization import AmpNativeOptimizer
from atorch.distributed.distributed import local_rank, parallel_group


@contextmanager
def amp_context(amp_config=None):
    if amp_config is not None:
        yield autocast(**amp_config)
    else:
        yield None


def _refresh_per_optimizer_state():
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def local_found_inf():
    counter = AutoAccelerateContext.counter
    num_stages_on_rank = len(AutoAccelerateContext.amp_native_pipe_grad_scaler[counter].values())
    condition = AutoAccelerateContext.grad_scaler_condition[counter]
    with condition:
        while AutoAccelerateContext.grad_scaler_counter[counter] < num_stages_on_rank:
            condition.wait()
        total_found_inf = sum(AutoAccelerateContext.grad_scaler_store[counter].values())
        # reset counter to 0 for next round
        AutoAccelerateContext.grad_scaler_counter[counter] = 0
    return total_found_inf


# FIXME Fails in Interleaved Mode
# If ranks have different number of stages, each ranks could have different number of all_reduce calls,
# and thus the process hangs. See #97
class PipeGradScaler(GradScaler):
    def __init__(
        self,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        process_group="pipe",
        stage_id=None,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.stage_id = stage_id
        self.process_group = process_group
        # non-driver ranks needs manual initialization
        if torch.distributed.is_initialized() and torch.cuda.is_available():
            self.device = f"cuda:{local_rank()}"
            self._lazy_init_scale_growth_tracker(self.device)
        self._maybe_init_process_group()

    def _maybe_init_process_group(self):
        if isinstance(self.process_group, str) and parallel_group(self.process_group) is not None:
            self.process_group = parallel_group(self.process_group)

    def _sync_inf_status(self, found_inf_per_device):
        if isinstance(self.process_group, str):
            self._maybe_init_process_group()
        found_inf = torch.tensor([sum(v.item() for v in found_inf_per_device.values())], device=self.device)
        counter = AutoAccelerateContext.counter
        AutoAccelerateContext.grad_scaler_store[counter].update({self.stage_id: found_inf})
        # Make sure all stage executors correctly stores the found_inf into rank's global store

        condition = AutoAccelerateContext.grad_scaler_condition[counter]
        with condition:
            # stage i finish found_inf
            AutoAccelerateContext.grad_scaler_counter[counter] += 1
            condition.notify()

        # Get the local sum first
        found_inf = local_found_inf()
        dist.all_reduce(found_inf, op=dist.ReduceOp.SUM, group=self.process_group)
        found_inf = found_inf[0]
        return found_inf

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        if not (self._sync_inf_status(optimizer_state["found_inf_per_device"])).item():
            retval = optimizer.step(*args, **kwargs)
        return retval

    def step(self, optimizer, *args, **kwargs):
        # must override this for optimizers that has customized scaling handling logics
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        retval = None
        if hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling:
            kwargs_ = kwargs
            has_grad_scaler_kwarg = "grad_scaler" in inspect.signature(optimizer.step).parameters
            if has_grad_scaler_kwarg:
                warnings.warn(
                    "GradScaler is going to stop passing itself as a keyword argument to the passed "
                    "optimizer. In the near future GradScaler registers `grad_scale: Tensor` and "
                    "`found_inf: Tensor` to the passed optimizer and let the optimizer use them directly.",
                    FutureWarning,
                )
                kwargs_.update({"grad_scaler": self})
            else:
                scaler = self._get_scale_async()
                found_inf = self._sync_inf_status(self._check_inf_per_device(optimizer))
                optimizer.grad_scale = None if optimizer_state["stage"] == OptState.UNSCALED else scaler
                optimizer.found_inf = found_inf
            retval = optimizer.step(*args, **kwargs_)
            optimizer_state["stage"] = OptState.STEPPED
            if not has_grad_scaler_kwarg:
                del optimizer.grad_scale
                del optimizer.found_inf
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

        retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval

    def update(self, new_scale=None):
        # update logic is tied with found_inf, so override this
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            counter = AutoAccelerateContext.counter
            AutoAccelerateContext.grad_scaler_store[counter].update({self.stage_id: found_inf_combined})
            # Make sure all stage executors correctly stores the found_inf into rank's global store
            condition = AutoAccelerateContext.grad_scaler_condition[counter]
            with condition:
                # stage i finish found_inf
                AutoAccelerateContext.grad_scaler_counter[counter] += 1
                condition.notify()
            # Get the local sum first
            found_inf_combined = local_found_inf()
            dist.all_reduce(found_inf_combined, op=dist.ReduceOp.SUM, group=self.process_group)

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = collections.defaultdict(_refresh_per_optimizer_state)


# FIXME 2 problems for interleaved mode now:
# 1. _hack_pipe_amp_optimizer will be called once only, so all stages on one rank will have the same grad scaler,
#   having multiple stage optimizers calling step/update method async leads to check inf bug
# 2. If Multiple grad scaler on a single rank, all-reducing is NOT the global result.
# FIX: Since all grad scalers are synced, it does not matter to use which one of them.
def scale_backward_wrapper(casted_loss):
    grad_scaler = next(iter(AutoAccelerateContext.amp_native_pipe_grad_scaler[AutoAccelerateContext.counter].values()))
    if isinstance(casted_loss, collections.abc.Sequence):
        loss = casted_loss[0]
    else:
        loss = casted_loss
    # To be compatible with FakeTensor shape inference
    if loss.is_meta:
        return casted_loss
    scaled_loss = grad_scaler.scale(loss)
    if isinstance(casted_loss, collections.abc.Sequence):
        return scaled_loss, *casted_loss[1:]
    else:
        return scaled_loss


def _hack_pipe_amp_optimizer():
    if PipeStageExecutor is None:
        return

    # hack PiPPy
    def instantiate_amp_optimizer(executor, optim_class, *args, **kwargs):
        stage_id = executor.stage_id
        grad_scaler = PipeGradScaler(process_group="pipe", stage_id=stage_id)
        counter = AutoAccelerateContext.counter

        if not hasattr(AutoAccelerateContext, "grad_scaler_counter"):
            AutoAccelerateContext.add_ac_attr("grad_scaler_counter", {counter: 0})
        elif counter not in AutoAccelerateContext.grad_scaler_counter:
            AutoAccelerateContext.grad_scaler_counter[counter] = 0

        if not hasattr(AutoAccelerateContext, "grad_scaler_condition"):
            AutoAccelerateContext.add_ac_attr("grad_scaler_condition", {counter: threading.Condition()})
        elif counter not in AutoAccelerateContext.grad_scaler_condition:
            AutoAccelerateContext.grad_scaler_condition[counter] = threading.Condition()

        if not hasattr(AutoAccelerateContext, "grad_scaler_store"):
            AutoAccelerateContext.add_ac_attr("grad_scaler_store", {counter: {stage_id: 0}})
        else:
            if counter in AutoAccelerateContext.grad_scaler_store:
                AutoAccelerateContext.grad_scaler_store[counter].update({stage_id: 0})
            else:
                AutoAccelerateContext.grad_scaler_store.update({counter: {stage_id: 0}})

        if not hasattr(AutoAccelerateContext, "amp_native_pipe_grad_scaler"):
            AutoAccelerateContext.add_ac_attr("amp_native_pipe_grad_scaler", {counter: {stage_id: grad_scaler}})
        else:
            if counter in AutoAccelerateContext.amp_native_pipe_grad_scaler:
                AutoAccelerateContext.amp_native_pipe_grad_scaler[counter].update({stage_id: grad_scaler})
            else:
                AutoAccelerateContext.amp_native_pipe_grad_scaler.update({counter: {stage_id: grad_scaler}})
        AutoAccelerateContext.add_ac_attr(
            "amp_native_pipe_grad_scaler", {AutoAccelerateContext.counter: {stage_id: grad_scaler}}
        )
        assert executor._should_instantiate_optim()
        with executor.optim_init_cv:
            optimizer = optim_class(executor.mod.parameters(), *args, **kwargs)
            executor.optimizer = AmpNativeOptimizer(optimizer, grad_scaler)
            executor.optim_init_cv.notify()
        return executor.optimizer

    setattr(PipeStageExecutor, "instantiate_optimizer", instantiate_amp_optimizer)
