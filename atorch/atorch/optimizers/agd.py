from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

__all__ = ("AGD",)


class AGD(torch.optim.Optimizer):
    r"""AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference as Preconditioning Matrix.
    Arguments:
        params (Params): Collection of parameters to be optimized,
            or an iterable of dictionaries specifying separate groups.
        lr (float, optional): The learning rate. Default is 1e-3.
        betas (tuple of 2 floats, optional): Coefficients used for computing running averages of gradient
            and its square. Default is (0.9, 0.999).
        delta (float, optional): Small constant for numerical stability to prevent division by zero. Default is 1e-5.
        weight_decay (float, optional): Weight decay coefficient. Default is 0.0.
        weight_decouple (bool, optional): If set to True, use decoupled weight decay. Default is True.
        fixed_decay (bool, optional): Enables fixed weight decay irrespective of the learning rate.
            Default setting is False.
        amsgrad (bool, optional): Applies the AMSGrad variant of the optimizer. Default is False.
        win (bool, optional): Applies the Win variant of the optimizer. Default is False.
        clip (bool, optional): Total update clip to prevent abnormal updates. Default is None.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        delta: float = 1e-5,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        amsgrad: bool = False,
        win: bool = False,
        clip: float = None,
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

        defaults = dict(
            lr=lr,
            betas=betas,
            delta=delta,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            amsgrad=amsgrad,
            win=win,
            clip=clip,
        )
        super(AGD, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
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

                if not group["win"]:
                    if group["weight_decouple"]:
                        if not group["fixed_decay"]:
                            p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                        else:
                            p.data.mul_(1.0 - group["weight_decay"])
                    else:
                        if group["weight_decay"] != 0:
                            grad.add_(p.data, alpha=group["weight_decay"])

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["win"]:
                        state["z"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                )

                state["step"] += 1
                exp_avg_old = exp_avg.detach().clone()
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                bias_correction1_old = 1 - beta1 ** (state["step"] - 1)
                bias_correction1, bias_correction2 = (
                    1 - beta1 ** state["step"],
                    1 - beta2 ** state["step"],
                )
                update = (
                    exp_avg * (1 / bias_correction1)
                    if state["step"] == 1
                    else exp_avg * (1 / bias_correction1) - exp_avg_old * (1 / bias_correction1_old)
                )
                exp_avg_sq.mul_(beta2).addcmul_(update, update, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    update = max_exp_avg_sq.sqrt()
                else:
                    update = exp_avg_sq.sqrt()

                delta_adjust = group["delta"] * bias_correction2.sqrt()
                update.clamp_(min=delta_adjust.to(update.device))

                lr_adjust = group["lr"] * bias_correction2.sqrt() / bias_correction1
                update = exp_avg / update
                if group["clip"] is not None:
                    update.clamp_(min=-group["clip"], max=group["clip"])
                if not group["win"]:
                    p.data.add_(update, alpha=-lr_adjust)
                else:
                    z = state["z"]
                    z.data.add_(update, alpha=-lr_adjust).mul_(1.0 / (1.0 + group["weight_decay"] * lr_adjust))
                    lr_adjust2 = 2 * lr_adjust
                    tao = 1.0 / (3.0 + lr_adjust2 * group["weight_decay"])
                    p.data.mul_(tao).add_(update, alpha=-tao * lr_adjust2).add_(z, alpha=2 * tao)
        return loss
