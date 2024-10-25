from typing import Optional

import torch
import torch.distributed as dist

from .base import TensorReducer


class LinearReducer(TensorReducer):
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        normalize: bool = True,
        weight_softmax_temperature: Optional[float] = None,
    ):
        super().__init__(process_group, weight_softmax_temperature)
        # use all gather based softmax in case a softmax temperature is given
        self.normalize = normalize and self.weight_softmax_temperature is None

    def _reduce_tensor(self, tensor: torch.Tensor, **kwargs):
        weight = self._refine_weight(tensor.device, tensor.dtype, **kwargs)

        with torch.no_grad():
            tensor *= weight
            if self.normalize:
                if weight is not None:
                    divisor = weight.clone()
                    dist.all_reduce(divisor, op=dist.ReduceOp.SUM, group=self.process_group)
                    divisor[divisor.abs() < 1e-8] = 1.0
                else:
                    divisor = dist.get_world_size(group=self.process_group)

                tensor /= divisor
            dist.all_reduce(tensor, group=self.process_group)
