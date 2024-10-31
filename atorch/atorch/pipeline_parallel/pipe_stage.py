import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import distributed as dist
from torch import nn

from atorch.communication.pipe_communicator import PipeCommunicator
from atorch.distributed.distributed import pipe_next_rank, pipe_prev_rank

logger = logging.getLogger(__name__)

try:
    from torch.distributed.pipelining._debug import map_debug_info
    from torch.distributed.pipelining._utils import flatten_args
    from torch.distributed.pipelining.stage import _PipelineStageBase
except (ImportError, ModuleNotFoundError):
    _PipelineStageBase = object
    map_debug_info, flatten_args = None, None


class PipeStage(_PipelineStageBase):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        io_mapping: Dict,
        group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, **kwargs)
        self.io_mapping = io_mapping
        # TODO: allocate inputs/outputs tensors in advance
        self.pipe_communicator = PipeCommunicator()

        self.prev_stage_rank = pipe_prev_rank()
        self.next_stage_rank = pipe_next_rank()

        self.fwd_received_tensors: Dict[int, List[torch.Tensor]] = {}
        self.bwd_received_grads: Dict[int, List[torch.Tensor]] = {}

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ):
        pass

    def _prepare_forward_infra(self, num_microbatches: int) -> None:
        pass

    def _prepare_backward_infra(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        if self.is_last:
            return []

        meta_id = f"fwd_send_stage{self.stage_index}_fwdchunk{fwd_chunk_id}"
        output = self.output_chunks[fwd_chunk_id]
        return self.pipe_communicator.send_ops_to_batch(output, target=self.next_stage_rank, meta_id=meta_id)

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        if self.is_first:
            return []

        meta_id = f"fwd_recv_stage{self.stage_index}_fwdchunk{fwd_chunk_id}"
        ops, fwd_received_tensors = self.pipe_communicator.recv_ops_to_batch(src=self.prev_stage_rank, meta_id=meta_id)
        self.fwd_received_tensors[fwd_chunk_id] = fwd_received_tensors
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        if not self.has_backward or self.is_first:
            return []

        meta_id = f"bwd_send_stage{self.stage_index}_bwdchunk{bwd_chunk_id}"
        return self.pipe_communicator.send_ops_to_batch(self.grads_input, target=self.prev_stage_rank, meta_id=meta_id)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        if not self.has_backward or self.is_last:
            return []

        meta_id = f"bwd_recv_stage{self.stage_index}_bwdchunk{bwd_chunk_id}"
        ops, bwd_received_grads = self.pipe_communicator.recv_ops_to_batch(src=self.next_stage_rank, meta_id=meta_id)
        self.bwd_received_grads[bwd_chunk_id] = bwd_received_grads
        return ops

    def _arrange_first_stage_input(self, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None):
        batch_mapping = self.io_mapping[1]
        if batch_mapping is None:
            return args, kwargs
        elif len(batch_mapping) > 0:
            new_args = []
            new_kwargs = {}
            kwargs_start = len(args)
            kwargs_end = len(args) + len(kwargs)
            idx_to_key = {}
            for i, key in enumerate(kwargs.keys()):
                idx_to_key[i + kwargs_start] = key

            for idx, bm in enumerate(batch_mapping):
                if isinstance(bm[1], int):
                    if idx < kwargs_start:
                        new_args.append(args[bm[1]])
                    else:
                        old_key = idx_to_key[bm[1]]
                        new_kwargs[bm[0]] = kwargs[old_key]
                elif isinstance(bm[1], str):
                    assert idx >= kwargs_start and idx <= kwargs_end
                    new_kwargs[bm[0]] = kwargs[bm[1]]
                else:
                    raise NotImplementedError(f"Type {type(bm[1])} is not support in batch mapping.")
            return tuple(new_args), new_kwargs
        elif len(batch_mapping) == 0:
            raise ValueError("batch mapping can't be empty list")

    def _filter_fwd_output(self, outputs: Any):
        activation_mapping = self.io_mapping[0]
        if activation_mapping is None:
            return outputs
        elif isinstance(outputs, torch.Tensor):
            for am in activation_mapping:
                assert isinstance(am[1], int)
                if am[1] == 0:
                    return outputs
                else:
                    return None
        else:
            new_outputs = []
            for am in activation_mapping:
                assert isinstance(am[1], int)
                new_outputs.append(outputs[am[1]])
        return tuple(new_outputs)

    def _validate_fwd_input(self, args, kwargs):
        pass

    def _validate_fwd_outputs(self, outputs: Tuple[torch.Tensor, ...]):
        pass

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage. They
        applies only to the first stage in most cases.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            args, kwargs = self._arrange_first_stage_input(args, kwargs)
            composite_args = args
            composite_kwargs = kwargs or {}
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = tuple(self.fwd_received_tensors[fwd_chunk_id])
            composite_kwargs = {}

        self._validate_fwd_input(args, kwargs)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        if type(output) is list:
            # HACK: this is a hacky workaround for the fact that export creates
            # output in list format
            output = tuple(output)

        # Filter output according to io mapping
        output = self._filter_fwd_output(output)

        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        # Prepare for final output merge or reduction
        self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            f"{self.log_prefix} Forwarded chunk {fwd_chunk_id}, outputs: {map_debug_info(output)}"  # noqa: G004
        )
        self._validate_fwd_outputs(output_tuple)

        return output

    def backward_one_chunk(self, bwd_chunk_id: int, loss=None, full_backward: bool = True):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self.bwd_received_grads[bwd_chunk_id]
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        self.grads_input = self.backward_maybe_with_nosync(bwd_kwargs)
        logger.debug(f"{self.log_prefix} Backwarded chunk {bwd_chunk_id}")  # noqa: G004

        return self.grads_input

    def batch_p2p(self, p2p_ops: List[torch.distributed.P2POp], desc: str):
        desc_str = f"{desc}, " if desc else ""
        logger.debug(f"batch_p2p {desc_str}{p2p_ops}")  # noqa: G004
        return self.pipe_communicator.batch_p2p(p2p_ops)
