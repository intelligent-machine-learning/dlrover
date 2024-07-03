from typing import Optional

import torch
import torch.distributed as dist
from typing_extensions import Literal

from .base import TensorReducer
from .sparsify import SparsificationMethod, sparsify


def get_consensus_mask_distributed(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
    process_group: dist.ProcessGroup = None,
):
    if mask_dtype is None:
        mask_dtype = delta.dtype

    delta_copy = delta.clone().to(delta.device)
    if method == "sum":
        dist.all_reduce(delta_copy, op=dist.ReduceOp.SUM, group=process_group)
        majority_sign = (delta_copy >= 0).to(mask_dtype) * 2 - 1
    elif method == "count":
        sign_copy = delta.sign().to(delta.device)
        dist.all_reduce(sign_copy, op=dist.ReduceOp.SUM, group=process_group)
        majority_sign = (sign_copy >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    sign = delta.sign().to(mask_dtype)
    return sign == majority_sign


class GTAReducer(TensorReducer):
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        consensus_method: Optional[Literal["sum", "count"]] = None,
        sparsification_method: Optional[SparsificationMethod] = None,
        normalize: bool = True,
        density: float = 1.0,
        int8_mask: bool = False,
        weight_softmax_temperature: Optional[float] = None,
    ):
        super().__init__(process_group, weight_softmax_temperature)
        self.concensus_method = consensus_method
        self.sparsification_method = sparsification_method
        self.density = density
        self.int8_mask = int8_mask
        # for gta is still make sense to do the post normalization even after softmax weight normalization
        self.normalize = normalize

    # since we require the computation to be done inplace, restrict all computation over "tensor"
    def _reduce_tensor(self, tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            # Optionally sparify the tensor
            if self.sparsification_method:
                tensor = sparsify(tensor, density=self.density, method=self.sparsification_method)

            weight = self._refine_weight(tensor.device, tensor.dtype, **kwargs)
            tensor *= weight

            tensor_dtype = tensor.dtype
            # apply the sign mask
            if self.concensus_method:
                mask_dtype = torch.int8 if self.int8_mask else tensor.dtype
                mask = get_consensus_mask_distributed(
                    tensor,
                    method=self.concensus_method,
                    mask_dtype=mask_dtype,
                    process_group=self.process_group,
                )
                tensor *= mask
                # transform tensor back to tensor_dtype
                tensor = tensor.to(tensor_dtype)

            # compute the normalization divisor
            # this induce another allreduce operation. A bit inefficient
            if self.normalize:
                if self.concensus_method:
                    divisor = mask * weight
                    dist.all_reduce(divisor, op=dist.ReduceOp.SUM, group=self.process_group)
                else:
                    divisor = weight
                    dist.all_reduce(divisor, op=dist.ReduceOp.SUM, group=self.process_group)

                divisor[divisor.abs() < 1e-8] = 1.0
                tensor /= divisor

            # allreduce after normalization
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
            tensor = tensor.to(tensor_dtype)
