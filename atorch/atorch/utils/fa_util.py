import functools
from functools import lru_cache

import torch
from torch.cuda.amp.autocast_mode import _cast, autocast


# patch fn to handle autocast
def _cast_fa_fn(fa_fn):
    @functools.wraps(fa_fn)
    def new_fa_fn(*args, **kwargs):
        if torch.is_autocast_enabled():
            cur_dtype = torch.get_autocast_gpu_dtype()
            with autocast(enabled=False):
                return fa_fn(*_cast(args, cur_dtype), **_cast(kwargs, cur_dtype))
        else:
            return fa_fn(*args, **kwargs)

    return new_fa_fn


@lru_cache()
def patch_fa_interface_to_autocast(interface):
    fn_names = [i for i in dir(interface) if i.startswith("flash_attn_") and i.endswith("_func")]
    for fn_name in fn_names:
        new_fa_fn = _cast_fa_fn(getattr(interface, fn_name))
        setattr(interface, fn_name, new_fa_fn)
