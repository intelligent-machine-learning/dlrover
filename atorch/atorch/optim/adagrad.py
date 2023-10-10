import torch

fused_enable = True
try:
    from apex.optimizers import Adagrad as AdagradImpl
except ImportError:
    fused_enable = False
    from torch.optim import Adagrad as AdagradImpl


class Adagrad(torch.optim.Optimizer):
    """Implements Adagrad algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
    ):

        if fused_enable:
            if lr_decay != 0:
                raise ValueError("Nonzero lr_decay {} is unsupported in fused" "Atorch".format(lr_decay))
            if initial_accumulator_value != 0:
                raise ValueError(
                    "Nonzero initial_accumulator_value {} is unsupported in"
                    "fused ATorch".foramt(initial_accumulator_value)
                )
            self._core = AdagradImpl(params=params, lr=lr, eps=eps, weight_decay=weight_decay)
        else:
            self._core = AdagradImpl(
                params=params,
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
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
