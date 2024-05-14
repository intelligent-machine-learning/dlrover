import torch
import torch.distributed as dist

from .base import TensorReducer


class LinearReducer(TensorReducer):
    def __init__(self, process_group: dist.ProcessGroup, normalize: bool = True):
        super().__init__(process_group)
        self.normalize = normalize

    def _reduce_tensor(self, tensor: torch.Tensor, **kwargs):
        weight = kwargs.get("weight", None)
        if weight is not None:
            weight = torch.tensor(weight, device=tensor.device)
        with torch.no_grad():
            if weight is not None:
                tensor *= weight
            if self.normalize:
                if weight is not None:
                    divisor = weight.clone()
                    dist.all_reduce(divisor, op=dist.ReduceOp.SUM, group=self.process_group)
                    divisor[divisor.abs() < 1e-12] = 1.0
                else:
                    divisor = dist.get_world_size(group=self.process_group)

                tensor /= divisor
            dist.all_reduce(tensor, group=self.process_group)
