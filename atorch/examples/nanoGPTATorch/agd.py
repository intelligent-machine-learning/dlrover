#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

__all__ = ("AGD",)


class AGD(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        delta: float = 1e-5,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        win: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, delta=delta, weight_decay=weight_decay, amsgrad=amsgrad, win=win)
        super(AGD, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = "AGD does not support sparse gradients."
                    raise RuntimeError(msg)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["win"]:
                        state["z"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["z"].add_(p.data.clone(), alpha=1)

                exp_avg, exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                )

                state["step"] += 1
                exp_avg_old = exp_avg.detach().clone()
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                bias_correction1_old = 1 - beta1 ** (state["step"] - 1)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                h_t = exp_avg / bias_correction1
                if state["step"] > 1:
                    h_t -= exp_avg_old / bias_correction1_old
                exp_avg_sq.mul_(beta2).addcmul_(h_t, h_t, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt()
                else:
                    denom = exp_avg_sq.sqrt()

                delta_adjust = group["delta"] * np.sqrt(bias_correction2)

                if torch.cuda.is_available():
                    denom = torch.where(denom > delta_adjust, denom, torch.Tensor([delta_adjust]).cuda())
                else:
                    denom = np.where(denom > delta_adjust, denom, delta_adjust)

                lr_adjust = group["lr"] * np.sqrt(bias_correction2) / bias_correction1
                weight_decay = group["weight_decay"]
                if not group["win"]:
                    p.data.mul_(1 - group["lr"] * weight_decay)
                    p.data.addcdiv_(exp_avg, denom, value=-lr_adjust)
                else:
                    z = state["z"]
                    z.addcdiv_(exp_avg, denom, value=-lr_adjust)
                    z.mul_(1.0 / (1.0 + weight_decay * lr_adjust))
                    lr_adjust2 = 2 * lr_adjust
                    tao = 1.0 / (3.0 + lr_adjust2 * weight_decay)
                    p.data.mul_(tao)
                    p.data.addcdiv_(exp_avg, denom, value=-tao * lr_adjust2)
                    p.data.add_(z, alpha=2 * tao)
        return loss
