from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup


@dataclass
class PipeMetaData:
    """
    This class is used to store the metadata of a pipe.
    """

    tensor_shape: Tuple[int]
    tensor_dtype: torch.dtype


def _broadcast_tensor_list(
    tensor_list: List[torch.Tensor],
    src: int,
    group: ProcessGroup,
    device: Optional[Union[torch.device, str, int]] = None,
):
    pass


def _send(tensor: torch.Tensor, src: int, dst: int, group: ProcessGroup):
    pass


def _recv(src: int, dst: int, group: ProcessGroup):
    pass


class PipeCommunicator:
    _cache: Dict[str, PipeMetaData] = {}

    def recv_forward(self, tensor, use_cache=False, cache_id=None) -> torch.Tensor:
        pass

    def send_forward(self, tensors):
        pass

    def send_forward_recv_backward(self, tensor, tensor_shape) -> torch.Tensor:
        pass

    def send_backward(self, tensor):
        pass

    def send_backward_recv_forward(self, tensor, tensor_shape) -> torch.Tensor:
        pass

    def recv_backward(self, shape, dtype) -> torch.Tensor:
        pass

    def send_forward_backward_recv_forward_backward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def send_backward_recv_backward(self) -> torch.Tensor:
        pass
