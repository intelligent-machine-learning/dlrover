from enum import Enum

try:
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleInterleaved1F1B
except (ImportError, ModuleNotFoundError):
    Schedule1F1B = object
    ScheduleInterleaved1F1B = object

from atorch.pipeline_parallel.pipe_module import PipeModule, make_pipe_module


class PipeSchedulerType(Enum):
    Schedule1F1B = "Schedule1F1B"
    ScheduleInterleaved1F1B = "ScheduleInterleaved1F1B"


def _init_pipe_schedule_from_pipe_module(pipe_module: PipeModule, sche_type, n_microbatches):
    if sche_type is PipeSchedulerType.Schedule1F1B:
        pipe_stage = pipe_module.stages[0]
        ScheduleClass = Schedule1F1B
    elif sche_type is PipeSchedulerType.ScheduleInterleaved1F1B:
        pipe_stage = pipe_module.stages
        ScheduleClass = ScheduleInterleaved1F1B
    else:
        raise NotImplementedError()

    # Attach to a schedule
    loss_fn = pipe_module.loss_func if pipe_module.loss_func is not None else pipe_module.ori_loss_func
    return ScheduleClass(pipe_stage, n_microbatches=n_microbatches, loss_fn=loss_fn)


def make_pipe_schedule(
    meta_model=None, model_provider=None, loss_func=None, strategy=None, distributed_context=None, config=None
):
    pipe_module = make_pipe_module(meta_model, model_provider, loss_func, strategy, distributed_context, config)

    return make_pipe_schedule_from_pipe_module(pipe_module, config)


def make_pipe_schedule_from_pipe_module(pipe_module: PipeModule, config=None):
    sche_type = PipeSchedulerType(config.sche_name)
    n_microbatches = config.n_microbatches
    return _init_pipe_schedule_from_pipe_module(pipe_module, sche_type, n_microbatches)
