from typing import Iterable, List, Union

from atorch.common.log_utils import default_logger as logger
from atorch.communication.communicator import PipeCommunicator
from atorch.utils.config import Config

from .pipe_module import PipeModule
from .scheduler import (
    PipeSchedulerType,
    _PipeState,
    forward_backward_no_pipelining_executor,
    one_forward_one_backward_executor,
    one_forward_one_backward_interleaving_executor,
)


class PipeEngine:
    """Pipeline Engine, a pipeline engine is responsible for scheduling the pipeline execution.
    It will execute the pipeline according to the scheduler configuration.

    Args:
        pipe_module (PipeModule): The pipeline module.
        config (Union[str, dict]): The configuration of the pipeline engine.

    Examples:
    model = make_pipe_module()
    engine = PipeEngine(model, config={
        "scheduler": "OneForwardOneBackwardInterleaving"
        "virtual_pp_size": 2,
        "pp_size": 4,
        "global_batchsize": 8,
        "micro_batchsize": 2,
    })
    """

    def __init__(self, pipe_module: PipeModule, config: Union[str, dict]):
        self.config = self._parse_config(config)
        self.model = pipe_module.to(self.config.device)
        self.pipe_state = _PipeState(self.config)
        self.pipe_communicator = PipeCommunicator()

    @classmethod
    def _parse_config(cls, input_config: Union[str, dict]):
        if isinstance(input_config, str):
            config = Config.from_file(input_config)
        elif isinstance(input_config, dict):
            config = Config(input_config)
        else:
            raise ValueError("Config must be a string or a dictionary")

        if not hasattr(config, "return_average_loss"):
            config.return_average_loss = False  # type: ignore
        if hasattr(config, "scheduler"):
            config.scheduler = PipeSchedulerType(config.scheduler)  # type: ignore
        else:
            logger.warning("Missing scheduler in PipeEngine config, set it to ForwardBackwardNoPipelinging.")
            config.scheduler = PipeSchedulerType.ForwardBackwardNoPipelinging  # to switch to 1F1B later.
        if not hasattr(config, "device"):
            config.device = "cuda"  # type: ignore
        if not hasattr(config, "virtual_pp_size"):
            config.virtual_pp_size = 1
        if not hasattr(config, "global_batchsize"):
            logger.warning("Missing global_batchsize in PipeEngine config, set it to 8.")
            config.global_batchsize = 8
        if not hasattr(config, "micro_batchsize"):
            logger.warning("Missing micro_batchsize in PipeEngine config, set it to 1.")
            config.micro_batchsize = 1

        return config

    _SCHEDULER_EXECUTOR_MAP = {
        PipeSchedulerType.ForwardBackwardNoPipelinging: forward_backward_no_pipelining_executor,
        PipeSchedulerType.OneForwardOneBackward: one_forward_one_backward_executor,
        PipeSchedulerType.OneForwardOneBackwardInterleaving: one_forward_one_backward_interleaving_executor,
    }

    def train_batch_step(self, data_iter: Union[Iterable, List[Iterable]]):
        if self.config.scheduler is PipeSchedulerType.OneForwardOneBackwardInterleaving:
            assert isinstance(data_iter, list)
            assert len(data_iter) == self.config.virtual_pp_size

        scheduler_executor = self._SCHEDULER_EXECUTOR_MAP[self.config.scheduler]
        forward_output = scheduler_executor(
            self.model, self.pipe_state, data_iter, self.pipe_communicator, self.config
        )  # type:ignore[operator]

        return forward_output
