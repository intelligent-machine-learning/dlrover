import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Union

import torch
import torch.distributed as dist


class TensorReducer(ABC):
    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group

    @abstractmethod
    def _reduce_tensor(self, tensor: torch.Tensor, **kwargs):
        """This is the part which actually implements the reducer logic
        The input tensor must be reduced inplace
        """
        ...

    def reduce_tensor(self, tensors: Union[Iterable[torch.Tensor], torch.Tensor], **kwargs):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]

        tensors_it1, tensors_it2 = itertools.tee(iter(tensors))
        flat_tensors = torch.cat([p.data.reshape(-1) for p in tensors_it1])
        self._reduce_tensor(flat_tensors, **kwargs)

        offset = 0
        for tensor in tensors_it2:
            tensor.data = flat_tensors[offset : offset + tensor.numel()].view_as(tensor).type_as(tensor)
            offset += tensor.numel()
