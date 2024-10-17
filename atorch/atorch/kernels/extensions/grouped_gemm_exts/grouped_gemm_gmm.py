import functools
from typing import Any, Callable, Optional, Tuple

import torch
from torch.cuda.amp.autocast_mode import _cast, autocast

from ..abstract_extension import AbstractExtension


# patch fn to handle autocast
@functools.lru_cache
def _cast_fn(fn):
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        if torch.is_autocast_enabled():
            cur_dtype = torch.get_autocast_gpu_dtype()
            with autocast(enabled=False):
                return fn(*_cast(args, cur_dtype), **_cast(kwargs, cur_dtype))
        else:
            return fn(*args, **kwargs)

    return new_fn


class GroupedGEMMExtension(AbstractExtension):
    def is_available(self) -> bool:

        available = False
        try:
            import grouped_gemm  # noqa
            import torch

            available = torch.cuda.is_available()
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        gmm, _ = self._load_with_ext_package()
        return gmm

    def _load_with_ext_package(self) -> Optional[Tuple[Callable[..., Any], Any]]:
        import grouped_gemm as gg

        gmm = _cast_fn(gg.ops.gmm)
        return gmm, gg


gmm = GroupedGEMMExtension().load()
