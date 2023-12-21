import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor

from atorch.optimizers.low_bit.functional import vectorwise_dequant, vectorwise_quant
from atorch.optimizers.low_bit.optim.q_optimizer import LowBitOptimizer

__all__ = ["Q_AdamW"]

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


class Q_AdamW(LowBitOptimizer):
    r"""Implements low-bit AdamW algorithm.
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay coefficient (default: 1e-2).
        q_bits (int, optional): The number of bits used for quantization (default: 4).
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        q_bits: int = 4,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not (isinstance(q_bits, int) and 1 <= q_bits <= 8):
            raise ValueError("Invalid q_bits value: {}".format(q_bits))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults, q_bits)

    def step(self, closure: OptLossClosure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros((), dtype=torch.float, device=p.device)
                    self.init_qstate(p, "exp_avg")
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros((), dtype=torch.float, device=p.device)
                    self.init_qstate(p, "exp_avg_sq")

                # dequantize
                q_exp_avg = state["exp_avg"]
                exp_avg_q_overhead = state["exp_avg_qstate"]["overhead"]
                exp_avg_q_map = state["exp_avg_qstate"]["qmap"]
                exp_avg_q_enabled = (
                    self.override_q_enable[id(p)]
                    if id(p) in self.override_q_enable
                    else state["exp_avg_qstate"]["enable"]
                )
                exp_avg_q_metadata = self.get_qmetadata_by_state_name("exp_avg")

                if q_exp_avg.numel() <= 1:
                    q_exp_avg.data = exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
                elif exp_avg_q_enabled:
                    exp_avg_q_overhead.update(exp_avg_q_metadata)
                    exp_avg = vectorwise_dequant(q_exp_avg, qmap=exp_avg_q_map, shape=p.shape, **exp_avg_q_overhead)
                    exp_avg_q_overhead.clear()
                else:
                    exp_avg = q_exp_avg

                q_exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq_q_overhead = state["exp_avg_sq_qstate"]["overhead"]
                exp_avg_sq_q_map = state["exp_avg_sq_qstate"]["qmap"]
                exp_avg_sq_q_enabled = (
                    self.override_q_enable[id(p)]
                    if id(p) in self.override_q_enable
                    else state["exp_avg_sq_qstate"]["enable"]
                )
                exp_avg_sq_q_metadata = self.get_qmetadata_by_state_name("exp_avg_sq")

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

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # quantize
                if exp_avg_q_enabled:
                    qx, gen = vectorwise_quant(exp_avg, qmap=exp_avg_q_map, shape=p.shape, **exp_avg_q_metadata)
                    q_exp_avg.data = qx
                    exp_avg_q_overhead.update(gen)

                if exp_avg_sq_q_enabled:
                    qx, gen = vectorwise_quant(
                        exp_avg_sq, qmap=exp_avg_sq_q_map, shape=p.shape, **exp_avg_sq_q_metadata
                    )
                    q_exp_avg_sq.data = qx
                    exp_avg_sq_q_overhead.update(gen)

        return loss
