import math
from enum import Enum

from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler as get_scheduler_trans
from transformers.trainer_utils import ExplicitEnum, SchedulerType


class AtorchSchedulerType(ExplicitEnum):
    LOG_WARMUP_LINEAR_DECAY = "log_warmup_linear_decay"


def get_linear_schedule_with_log_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        inverse_log_warm_up = 1.0 / math.log(num_warmup_steps + 1e-6)
        if current_step == 0:
            return 0.0
        if current_step < num_warmup_steps:
            return inverse_log_warm_up * math.log(current_step + 1e-6)
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_ATORCHSCHEDULER_FUNCTION = {AtorchSchedulerType.LOG_WARMUP_LINEAR_DECAY: get_linear_schedule_with_log_warmup}

SCHEDULER_NAMES = list(SchedulerType._value2member_map_.keys())
ATORCHSCHEDULER_NAMES = list(AtorchSchedulerType._value2member_map_.keys())


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    if name in SCHEDULER_NAMES:
        return get_scheduler_trans(name, optimizer, num_warmup_steps, num_training_steps)
    elif name in ATORCHSCHEDULER_NAMES:
        name = AtorchSchedulerType(name)
        schedule_func = TYPE_TO_ATORCHSCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise ValueError(
            f"{name} is not a valid schedule name, please select one of {SCHEDULER_NAMES + ATORCHSCHEDULER_NAMES}."
        )


class AsyncCheckpointSignal(Enum):
    # Send signal to the manager process to delete checkpoint
    DELETE_CKPT = 1

    # Send signal to the manager process to terminate the saving process which is over.
    SAVE_OVER = 2

    # Send signal to the manager process to destroy itself at the end of training.
    TRAIN_OVER = 3


class PipeMessageEntity:
    def __init__(self, signal_type: AsyncCheckpointSignal, pid: int = None, ckpt_path: str = None):
        self.signal_type = signal_type
        self.pid = pid
        self.ckpt_path = ckpt_path
