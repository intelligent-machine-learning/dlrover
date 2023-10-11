from torch.optim.optimizer import Optimizer, required

fused_enable = True
try:
    from apex.optimizers import FusedSGD
except ImportError:
    fused_enable = False
    from torch.optim import SGD as RawSGD


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        if fused_enable:
            self._core = FusedSGD(
                params=params,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        else:
            self._core = RawSGD(
                params=params,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
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
