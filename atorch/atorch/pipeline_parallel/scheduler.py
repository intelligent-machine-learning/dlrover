import contextlib
from enum import Enum
from typing import Dict, Iterable, List, Optional, Union

import torch

from atorch.common.util_func import data_to_device
from atorch.communication.communicator import PipeCommunicator
from atorch.distributed import distributed as _distributed_context
from atorch.distributed.distributed import get_data_partition_rank_and_size
from atorch.pipeline_parallel.pipe_module import PipeModule
from atorch.utils.config import Config


class P2PType(Enum):
    Send = "Send"
    Recv = "Recv"


class PipeSchedulerType(Enum):
    ForwardBackwardNoPipelinging = "ForwardBackwardNoPipelining"
    OneForwardOneBackward = "OneForwardOneBackward"
    OneForwardOneBackwardInterleaving = "OneForwardOneBackwardInterleaving"


class _PipeState:
    def __init__(self, config: Config) -> None:
        """
        config contains: scheduler, virtual_pp_size, pp_size
        input_tensors: input tensors for corresponding model chunks
        output_tensors: output tensors or loss (for last stage) for corresponding model chunks
        output_tensor_grads: output tensor's grads for corresponding model chunks
        """
        self.config: Config = config
        self.input_tensors: Dict = {v_pp_rank: [] for v_pp_rank in range(self.config.virtual_pp_size)}
        self.output_tensors: Dict = {v_pp_rank: [] for v_pp_rank in range(self.config.virtual_pp_size)}
        self.output_tensor_grads: Dict = {v_pp_rank: [] for v_pp_rank in range(self.config.virtual_pp_size)}
        self.forward_output_store: List = []
        self.tensor_shapes_cache: Dict = {}
        self.num_micro_batches: int = self._cal_num_microbatches(self.config)
        self.cur_virtual_pp_rank: int = 0

    def reset_store(self):
        for v_rank in range(self.config.virtual_pp_size):
            self.input_tensors[v_rank] = []
            self.output_tensors[v_rank] = []
            self.output_tensor_grads[v_rank] = []
        self.forward_output_store = []
        self.cur_virtual_pp_rank = 0

    def empty_tensors(self, v_rank=None, empty_inputs=True, empty_outputs=True, empty_output_grads=True):
        rank = self.cur_virtual_pp_rank if v_rank is None else v_rank
        if empty_inputs:
            self.input_tensors[rank] = []
        if empty_outputs:
            self.output_tensors[rank] = []
        if empty_output_grads:
            self.output_tensor_grads[rank] = []

    @property
    def forward_output(self):
        return self.forward_output_store

    def get_input_tensors(self, v_rank=None):
        return self.input_tensors[self.cur_virtual_pp_rank if v_rank is None else v_rank]

    def get_output_tensors(self, v_rank=None):
        return self.output_tensors[self.cur_virtual_pp_rank if v_rank is None else v_rank]

    def get_output_tensor_grads(self, v_rank=None):
        return self.output_tensor_grads[self.cur_virtual_pp_rank if v_rank is None else v_rank]

    def append_input_tensor(self, tensor: Optional[torch.Tensor]):
        self.input_tensors[self.cur_virtual_pp_rank].append(tensor)

    def append_output_tensor(self, tensor: Optional[torch.Tensor]):
        self.output_tensors[self.cur_virtual_pp_rank].append(tensor)

    def append_output_tensor_grad(self, tensor: Optional[torch.Tensor]):
        self.output_tensor_grads[self.cur_virtual_pp_rank].append(tensor)

    def append_forward_output(self, tensor: Optional[torch.Tensor]):
        self.forward_output_store.append(tensor)

    def get_input_tensor(self, idx: int):
        return self.input_tensors[self.cur_virtual_pp_rank][idx]

    def get_output_tensor(self, idx: int):
        return self.output_tensors[self.cur_virtual_pp_rank][idx]

    def get_output_tensor_grad(self, idx: int):
        return self.output_tensor_grads[self.cur_virtual_pp_rank][idx]

    def _cal_num_microbatches(self, config: Config):
        _, dp_size = get_data_partition_rank_and_size()
        micro_batch_times_data_parallel = config.micro_batchsize * dp_size
        assert config.global_batchsize % micro_batch_times_data_parallel == 0

        return config.global_batchsize // micro_batch_times_data_parallel


def is_batch_on_device(batch, device):
    for data in batch:
        if data is not None and isinstance(batch[data], torch.Tensor) and batch[data].device != torch.device(device):
            return False
    return True


def get_model_inputs(model, pipe_state, batch_data):
    activation_mapping, batch_mapping = model.get_io_mapping(pipe_state.cur_virtual_pp_rank)

    activation_list_inputs = []
    activation_dict_inputs = {}
    batch_dict_inputs = {}

    input_tensors = pipe_state.get_input_tensors()
    if activation_mapping is None:
        activation_list_inputs += input_tensors
    else:
        for name_in_forward, idx in activation_mapping:
            activation_dict_inputs[name_in_forward] = input_tensors[idx]

    if batch_mapping is not None:
        for name_in_forward, name_in_batch in batch_mapping:
            batch_dict_inputs[name_in_forward] = batch_data[name_in_batch]

    return activation_list_inputs, activation_dict_inputs, batch_dict_inputs


def _get_batch(data_iter: Iterable, device: Union[str, int, torch.device]):
    batch = next(data_iter)  # type: ignore
    if not is_batch_on_device(batch, device):
        batch = data_to_device(batch, device)

    return batch


def _forward_step(
    model: PipeModule,
    data_iter: Iterable,
    pipe_state: _PipeState,
    num_microbatches: int,
    config: Config,
):
    batch_data = _get_batch(data_iter, config.device) if model.requires_batch_input() else [None]
    input_list, act_input_dict, batch_input_dict = get_model_inputs(model, pipe_state, batch_data)
    output_tensor, loss_func = model(*input_list, **act_input_dict, **batch_input_dict)

    if _distributed_context.is_pipe_last_stage():
        output_tensor = loss_func(batch_input_dict, output_tensor)
        # loss averaging behavior (ie, over the number of microbatches)
        output_tensor = output_tensor / num_microbatches

        # Reduce loss across dp groups
        if config.return_average_loss:
            _, dp_size = get_data_partition_rank_and_size()
            if dp_size is not None and dp_size > 1:
                reporting_loss = output_tensor.clone().detach()
                torch.distributed.all_reduce(reporting_loss, group=_distributed_context.parallel_group("data"))
                pipe_state.append_forward_output(reporting_loss.item())
        else:
            pipe_state.append_forward_output(output_tensor.item())

    return output_tensor


def _backward_step(model: PipeModule, pipe_state: _PipeState, config: Config):
    input_tensors = pipe_state.get_input_tensors()
    output_tensors = pipe_state.get_output_tensors()
    output_tensor_grads = pipe_state.get_output_tensor_grads()
    for tensor in input_tensors:
        if tensor.requires_grad:
            tensor.retain_grad()

    # TODO: grad scale
    # TODO: megatron backward for `deallocate_pipeline_outputs` optimization
    if _distributed_context.is_pipe_last_stage():
        output_tensors[0].backward()
    else:
        # output_tensor_grad is received from next stage in 1F1B
        torch.autograd.backward(output_tensors, grad_tensors=output_tensor_grads)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [x.grad if x.requires_grad else None for x in input_tensors]
    return input_tensor_grad


def one_forward_one_backward_executor(
    model: PipeModule,
    pipe_state: _PipeState,
    data_iter: Union[Iterable, List[Iterable]],
    pipe_communicator: PipeCommunicator,
    config: Config,
) -> List[torch.Tensor]:
    """
    1F1B non-interleaved scheduler executor.
    """
    pass


def forward_backward_no_pipelining_executor(
    model: PipeModule,
    pipe_state: _PipeState,
    data_iter: Union[Iterable, List[Iterable]],
    pipe_communicator: PipeCommunicator,
    config: Config,
) -> List[torch.Tensor]:
    assert len(model.modules) == 1
    if isinstance(data_iter, list):
        assert len(data_iter) == 1
        data_iter = data_iter[0]

    pipe_state.reset_store()
    num_microbatches = pipe_state.num_micro_batches
    with contextlib.nullcontext():
        for i in range(num_microbatches - 1):
            output_tensor = _forward_step(model, data_iter, pipe_state, num_microbatches, config)
            pipe_state.append_output_tensor(output_tensor)
            _backward_step(model, pipe_state, config)
            pipe_state.empty_tensors()  # should we put it in backward?

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = _forward_step(model, data_iter, pipe_state, num_microbatches, config)
    pipe_state.append_output_tensor(output_tensor)
    _backward_step(model, pipe_state, config)

    return pipe_state.forward_output


def one_forward_one_backward_interleaving_executor(
    model: PipeModule,
    pipe_state: _PipeState,
    data_iter: Union[Iterable, List[Iterable]],
    pipe_communicator: PipeCommunicator,
    config: Config,
) -> List[torch.Tensor]:
    pass
