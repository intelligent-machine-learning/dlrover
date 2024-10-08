import functools
from typing import Any, Callable, Optional

from atorch.kernels.extensions.grouped_gemm_exts.grouped_gemm_gmm import _cast_fn

from ..abstract_extension import AbstractExtension


# patch fn to handle autocast
@functools.lru_cache
def _convert_fn(fn):
    @functools.wraps(fn)
    def new_fn(a, b, batch_sizes, trans_b):
        x = a
        weight = b.transpose(1, 2) if trans_b else b
        # size of groupList can not be 1.If expected group num is 1, groupList should be nullptr.
        if len(batch_sizes) == 1:
            return x @ weight
        from itertools import accumulate

        group_list = list(accumulate(batch_sizes.tolist()))
        group_type = 0
        return fn(x, weight, bias=None, group_list=group_list, group_type=group_type)

    return new_fn


class GroupedGEMMNpuExtension(AbstractExtension):
    def is_available(self) -> bool:

        available = False
        try:
            from mindspeed.ops import gmm

            available = hasattr(gmm, "npu_gmm")
        except (ImportError, ModuleNotFoundError):
            available = False
        return available

    def load(self) -> Optional[Callable[..., Any]]:
        if not self.is_available():
            return None

        from mindspeed.ops import gmm

        gmm = _cast_fn(_convert_fn(gmm.npu_gmm))
        return gmm


npu_gmm = GroupedGEMMNpuExtension().load()
