import math

import torch
from torch.optim.optimizer import Optimizer


class AdamL(Optimizer):
    r"""Implements Adam with Lasso algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        l1 (float, optional): l1 regularization coefficient (default: 0)
        l21 (float, optional): l21 regularization coefficient (default: 0)
        adjust_l1 (bool, optional): using the Euclidean norm of l1 regularization
            if set False (default: True)

    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, l1=0.0, l21=0.0, adjust_l1=True
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
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 value: {}".format(l1))
        if not 0.0 <= l21:
            raise ValueError("Invalid l21 value: {}".format(l21))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, l1=l1, l21=l21, adjust_l1=adjust_l1)
        super(AdamL, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

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

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamL does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                l1 = group["l1"]
                l21 = group["l21"]
                lr = group["lr"]
                adjust_l1 = group["adjust_l1"]
                if lr <= 0.0:
                    continue

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                linear = exp_avg / bias_correction1 - denom * p.detach() / lr

                if adjust_l1:
                    l1_adjust = l1 * denom.sqrt()
                    l1_reg_adjust = torch.clamp(linear, -l1_adjust, l1_adjust) if l1 > 0 else torch.zeros_like(linear)
                    l1_linear = l1_reg_adjust - linear
                else:
                    l1_reg_adjust = torch.clamp(linear, -l1, l1) if l1 > 0 else torch.zeros_like(linear)
                    l1_linear = l1_reg_adjust - linear

                if p.dim() > 1 and l21 > 0:
                    l21_norm = torch.norm((l1_linear / denom.sqrt()), p=2, dim=1, keepdim=True) / math.sqrt(p.size(1))
                    state["l21_norm"] = l21_norm
                    p.data.copy_(
                        torch.where(
                            l21_norm <= l21,
                            torch.zeros_like(p.data),
                            lr * l1_linear / (denom * (1 + lr * group["weight_decay"])) * (1.0 - l21 / l21_norm),
                        )
                    )
                else:
                    p.data.copy_(lr * l1_linear / (denom * (1 + lr * group["weight_decay"])))

        return loss
