from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor

from atorch.optimizers.low_bit.functional import vectorwise_dequant, vectorwise_quant
from atorch.optimizers.low_bit.optim.q_optimizer import LowBitOptimizer

__all__ = ["Q_CAME"]

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
State = Dict[str, Any]
OptFloat = Optional[float]
Eps2 = Tuple[float, float]
Betas3 = Tuple[float, float, float]
ParamGroup = Dict[str, Any]


class Q_CAME(LowBitOptimizer):
    r"""Implements low-bit CAME algorithm.
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: None).
        eps (Tuple[float, float], optional): Regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-16)).
        clip_threshold (float, optional): Threshold of root mean square of
            final gradient update (default: 1.0).
        betas (tuple[float, float, float]): Coefficient used for computing running averages of
            update, square gradient and instability (default: (0.9, 0.999, 0.9999))).
        weight_decay (float, optional): Weight decay coefficient (default: 1e-2).
        q_bits (int, optional): The number of bits used for quantization (default: 4).
    """

    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        eps: Eps2 = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: Betas3 = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        q_bits: int = 4,
    ):
        if lr is not None and not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps[0]:
            raise ValueError("Invalid epsilon value at index 0: {}".format(eps[0]))
        if not 0.0 <= eps[1]:
            raise ValueError("Invalid epsilon value at index 1: {}".format(eps[1]))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not (isinstance(q_bits, int) and 1 <= q_bits <= 8):
            raise ValueError("Invalid q_bits value: {}".format(q_bits))

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults, q_bits)

    def _get_options(self, param_shape: Tuple[int, ...]):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor: torch.Tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure: OptLossClosure = None):
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
                    raise RuntimeError("Q_CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = self._get_options(grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros((), dtype=torch.float, device=p.device)
                    self.init_qstate(p, "exp_avg")
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)

                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros((), dtype=torch.float, device=p.device)
                        self.init_qstate(p, "exp_avg_sq")

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(group["betas"][1]).add_(update.mean(dim=-1), alpha=1.0 - group["betas"][1])
                    exp_avg_sq_col.mul_(group["betas"][1]).add_(update.mean(dim=-2), alpha=1.0 - group["betas"][1])

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
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

                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

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

                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                if factored:
                    res = (update - exp_avg) ** 2 + group["eps"][1]
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    exp_avg_res_row.mul_(group["betas"][2]).add_(res.mean(dim=-1), alpha=1.0 - group["betas"][2])
                    exp_avg_res_col.mul_(group["betas"][2]).add_(res.mean(dim=-2), alpha=1.0 - group["betas"][2])

                    # Approximation of exponential moving average of instability
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["weight_decay"] * group["lr"])
                update.mul_(group["lr"])
                p.data.add_(-update)

                # quantize
                if exp_avg_q_enabled:
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
