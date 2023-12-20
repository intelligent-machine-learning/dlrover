import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor

from atorch.optimizers.low_bit.functional import vectorwise_dequant, vectorwise_quant
from atorch.optimizers.low_bit.optim.q_optimizer import LowBitOptimizer

__all__ = ["Q_Adafactor"]

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
State = Dict[str, Any]
OptFloat = Optional[float]

Eps2 = Tuple[float, float]
ParamGroup = Dict[str, Any]


class Q_Adafactor(LowBitOptimizer):
    r"""Implements low-bit Adafactor algorithm.
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: None).
        eps2 (Tuple[float, float], optional): Regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3)).
        clip_threshold (float, optional): Threshold of root mean square of
            final gradient update (default: 1.0).
        decay_rate (float, optional): Coefficient used to compute running averages of square
            gradient (default: -0.8).
        beta1 (float, optional): Coefficient used for computing running averages of gradient
            (default: None).
        weight_decay (float, optional): Weight decay coefficient (default: 1e-2).
        scale_parameter (bool, optional): If true, learning rate is scaled by root mean square
            of parameter (default: True).
        relative_step (bool, optional): If true, time-dependent learning rate is computed
            instead of external learning rate (default: True).
        warmup_init (bool, optional): Time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False).
        q_bits (int, optional): The number of bits used for quantization (default: 4).
    """

    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        eps2: Eps2 = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: OptFloat = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        q_bits: int = 4,
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not (isinstance(q_bits, int) and 1 <= q_bits <= 8):
            raise ValueError("Invalid q_bits value: {}".format(q_bits))

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults, q_bits)

    def _get_lr(self, param_group: ParamGroup, param_state: State) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group: ParamGroup, param_shape: Tuple[int, ...]) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(
        self,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Q_Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    # Exponential moving average of gradient values
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros((), dtype=torch.float, device=p.device)
                        self.init_qstate(p, "exp_avg")
                    # Exponential moving average of squared gradient values
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros((), dtype=torch.float, device=p.device)
                        self.init_qstate(p, "exp_avg_sq")

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps2"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square
                    # of gradient
                    self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                    update.mul_(grad)
                else:
                    q_exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq_q_overhead = state["exp_avg_sq_qstate"]["overhead"]
                    exp_avg_sq_q_map = state["exp_avg_sq_qstate"]["qmap"]
                    exp_avg_sq_q_enabled = (
                        self.override_q_enable[id(p)]
                        if id(p) in self.override_q_enable
                        else state["exp_avg_sq_qstate"]["enable"]
                    )
                    exp_avg_sq_q_metadata = self.get_qmetadata_by_state_name("exp_avg_sq")

                    # dequantize
                    if q_exp_avg_sq.numel() <= 1:
                        q_exp_avg_sq.data = exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)
                    elif exp_avg_sq_q_enabled:
                        exp_avg_sq_q_overhead.update(exp_avg_sq_q_metadata)
                        exp_avg_sq = vectorwise_dequant(
                            q_exp_avg_sq, qmap=exp_avg_sq_q_map, shape=p.shape, **exp_avg_sq_q_overhead
                        )
                        exp_avg_sq_q_overhead.clear()
                    else:
                        exp_avg_sq = q_exp_avg_sq

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(max(1.0, self._rms(update) / group["clip_threshold"]))
                update.mul_(lr)

                if use_first_moment:
                    q_exp_avg = state["exp_avg"]
                    exp_avg_q_overhead = state["exp_avg_qstate"]["overhead"]
                    exp_avg_q_map = state["exp_avg_qstate"]["qmap"]
                    exp_avg_q_enabled = (
                        self.override_q_enable[id(p)]
                        if id(p) in self.override_q_enable
                        else state["exp_avg_qstate"]["enable"]
                    )
                    exp_avg_q_metadata = self.get_qmetadata_by_state_name("exp_avg")

                    # dequantize
                    if q_exp_avg.numel() <= 1:
                        q_exp_avg.data = exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
                    elif exp_avg_q_enabled:
                        exp_avg_q_overhead.update(exp_avg_q_metadata)
                        exp_avg = vectorwise_dequant(q_exp_avg, qmap=exp_avg_q_map, shape=p.shape, **exp_avg_q_overhead)
                        exp_avg_q_overhead.clear()
                    else:
                        exp_avg = q_exp_avg

                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["weight_decay"] * lr)

                p.data.add_(-update)

                # quantize
                if use_first_moment and exp_avg_q_enabled:
                    qx, gen = vectorwise_quant(exp_avg, qmap=exp_avg_q_map, shape=p.shape, **exp_avg_q_metadata)
                    q_exp_avg.data = qx
                    exp_avg_q_overhead.update(gen)

                if not factored and exp_avg_sq_q_enabled:
                    qx, gen = vectorwise_quant(
                        exp_avg_sq, qmap=exp_avg_sq_q_map, shape=p.shape, **exp_avg_sq_q_metadata
                    )
                    q_exp_avg_sq.data = qx
                    exp_avg_sq_q_overhead.update(gen)

        return loss
