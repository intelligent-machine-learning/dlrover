import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import torch
import torch.distributed as dist


class TensorReducer(ABC):
    def __init__(self, process_group: dist.ProcessGroup, weight_softmax_temperature: Optional[float] = None):
        self.process_group = process_group
        self.weight_softmax_temperature = (
            weight_softmax_temperature
            if weight_softmax_temperature is None
            else torch.tensor(weight_softmax_temperature, dtype=torch.float32)
        )

    def _refine_weight(self, device, dtype, **kwargs):
        weight = kwargs.get("weight", 1.0)
        step_weight = kwargs.get("step_weight", 1.0)
        step_weight_ratio = kwargs.get("step_weight_ratio", 0.0)
        if self.weight_softmax_temperature is not None:
            with torch.cuda.amp.autocast(enabled=False):
                weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
                # Ensure the tensor is at least 1D
                if weight.dim() == 0:
                    weight = weight.unsqueeze(0)

                # to avoid numerical issue, use all gather to compute softmax
                weight /= self.weight_softmax_temperature.to(device)
                world_size = dist.get_world_size(self.process_group)
                all_weights = [torch.zeros_like(weight) for _ in range(world_size)]
                dist.all_gather(all_weights, weight, group=self.process_group)
                all_weights = torch.cat(all_weights)
                softmax_weights = torch.nn.functional.softmax(all_weights, dim=0)
                weight = softmax_weights[dist.get_rank(self.process_group)].to(dtype)
        else:
            weight = torch.tensor(weight, dtype=dtype, device=device)

        weight = weight * (1 - step_weight_ratio) + step_weight * step_weight_ratio
        weight = weight.to(dtype)

        return weight

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
