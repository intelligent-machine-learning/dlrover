import warnings

import torch
from torch.nn import Linear

try:

    # from apex.fused_dense import FusedDenseGeluDenseFunc  # TODO: https://github.com/NVIDIA/apex/issues/1605
    from apex.fused_dense.fused_dense import DenseNoBiasFunc, FusedDense, FusedDenseFunc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Using AmpFusedDense but no atorch/apex installed:%s" % e)
    FusedDense = object
    # raise error?


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())


def _fused_dense(input, weight, bias):
    args = _cast_if_autocast_enabled(input, weight, bias)
    with torch.cuda.amp.autocast(enabled=False):
        return FusedDenseFunc.apply(*args)


def _dense_no_bias(input, weight):
    args = _cast_if_autocast_enabled(input, weight)
    with torch.cuda.amp.autocast(enabled=False):
        return DenseNoBiasFunc.apply(*args)


class AmpFusedDense(FusedDense):
    """
    differ with FusedDense:
    1. fix input shape issue
    2. compatible with old version apex when using autocast
    """

    def forward(self, input):
        assert input.dim() >= 2  # TODO: 1d input have issue of number consistency
        *other_dims, h = input.size()
        input_ = input.reshape(-1, h)
        out_shape = other_dims + [self.out_features]

        if self.bias is not None:
            out = _fused_dense(input_, self.weight, self.bias)
        else:
            out = _dense_no_bias(input_, self.weight)

        return out.reshape(*out_shape)


def replace_linear(module, cur_name):
    # Linear __init__ : bias is True/False ,but self.bias is Tensor
    for name, child in module.named_children():
        child_name = cur_name + "." + name
        if isinstance(child, Linear):
            if child.bias is not None:
                new_module = AmpFusedDense(child.in_features, child.out_features, bias=True)
            else:
                new_module = AmpFusedDense(child.in_features, child.out_features, bias=False)
            new_module.load_state_dict(child.state_dict())
            setattr(module, name, new_module)
        else:
            replace_linear(child, child_name)
