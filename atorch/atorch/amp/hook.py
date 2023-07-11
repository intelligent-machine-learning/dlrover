from typing import Dict, List

from apex.amp import _initialize
from torch import Tensor

to_type_original = _initialize.to_type


def sample_list_to_type(dtype, t):
    """
    Hook `_initialize.to_type`. Original `to_type` only handle the case
    that `t` is a torch.Tensor. `sample_list_to_type` can also handle
    the case that t is a list or a dict.
    """
    if isinstance(t, Dict):
        for k, v in t.items():
            if isinstance(v, Tensor):
                if v.is_floating_point():
                    t[k] = v.to(dtype)
        return t
    elif isinstance(t, List):
        for i, elem in enumerate(t):
            if isinstance(elem, Tensor):
                if elem.is_floating_point():
                    t[i] = elem.to(dtype)
        return t
    else:
        return to_type_original(dtype, t)


_initialize.to_type = sample_list_to_type
