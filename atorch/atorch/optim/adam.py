import torch

fused_enable = True
try:
    from apex.optimizers import FusedAdam as AdamImpl
except ImportError:
    fused_enable = False
    from torch.optim import Adam as AdamImpl


class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lt (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coeficients used for computing
            runing averages of gradients and its square.(default:
            (0.9, 0.999))
        eps (float, optional):term added to the denominator to improv
            numberial stability. (1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): wether to use the AMSGrad variant of thie
            algorithm from the paper `On the Convergence of Adam and
            Beyond`_ (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".foramt(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalia epsion value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parametrt at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if fused_enable:
            if amsgrad:
                raise RuntimeError("ATorch fused Adam does not support AMSGRad variant")
            self._core = AdamImpl(
                params=params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adam_w_mode=False,
            )
        else:
            self._core = AdamImpl(
                params=params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )

    @property
    def state(self):
        return self._core.state

    @property
    def param_groups(self):
        return self._core.param_groups

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self._core.param_groups):
            format_string += "\n"
            format_string += "Parameter Group {0}\n".format(i)
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += "    {0}: {1}\n".format(key, group[key])
        format_string += ")"
        return format_string

    def zero_grad(self, set_to_none: bool = False):
        if fused_enable:
            self._core.set_grad_none = set_to_none
            self._core.zero_grad()
        else:
            self._core.zero_grad(set_to_none)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self._core.step(closure)
        return loss

    def state_dict(self):
        state_dict = self._core.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self._core.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        self._core.add_param_group(param_group)
