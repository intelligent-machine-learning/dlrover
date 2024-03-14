import sys

import torch.distributed as dist
from torch.cuda.amp import GradScaler

from atorch.common.log_utils import default_logger as logger
from atorch.utils.version import torch_version

if torch_version() >= (1, 12, 0):  # type: ignore
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
else:
    from fairscale.optim.grad_scaler import ShardedGradScaler


class BF16GradScaler(GradScaler):
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self.overflow = False
        super().__init__(
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self._init_scale = 1.0
        self._backoff_factor = 1.0
        self._growth_factor = 1.0
        self._growth_interval = sys.maxsize

    def scale(self, outputs):
        self.overflow = False
        return super().scale(outputs)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        else:
            self.overflow = True
            logger.info("Found overflow. Skip optim.step!")
        return retval

    def has_overflow(self):
        return self.overflow


class BF16ShardedGradScaler(ShardedGradScaler):
    def __init__(
        self,
        init_scale=2.0**16,
        backoff_factor=0.5,
        growth_factor=2.0,
        growth_interval=2000,
        enabled=True,
        process_group=dist.group.WORLD,
    ):
        self.overflow = False
        super().__init__(
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
            process_group=process_group,
        )
        self._init_scale = 1.0
        self._backoff_factor = 1.0
        self._growth_factor = 1.0
        self._growth_interval = sys.maxsize

    def scale(self, outputs):
        self.overflow = False
        return super().scale(outputs)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        else:
            self.overflow = True
            logger.info("Found overflow. Skip optim.step!")
        return retval

    def has_overflow(self):
        return self.overflow
