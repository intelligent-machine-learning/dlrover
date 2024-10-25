from typing import List, Union

from torch.optim import lr_scheduler

from atorch.trainer.base.inferface import Stateful
from atorch.trainer.base.optimizer import AtorchOptimizer


class AtorchScheduler(lr_scheduler.LRScheduler, Stateful):  # type: ignore[name-defined]
    def __init__(self, scheduler, optimizers: Union[AtorchOptimizer, List[AtorchOptimizer]]):
        self.scheduler = scheduler
        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]

    @classmethod
    def from_config(cls, distributed_type, *args, **kwargs):
        pass

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()

    def step(self):
        pass

    def state_dict(self, *args, **kwargs):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass
