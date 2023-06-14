import math

import torch
import torch.nn.functional as F

# from atorch.common.log_utils import default_logger as logger


def one_hot(tensor, num_classes):
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    if torch.__version__ >= "1.9":
        return F.one_hot(tensor, num_classes=num_classes)
    else:
        assert num_classes > 0, "num_classes must be a positive integer"
        ret = torch.zeros(
            tensor.shape + (num_classes,),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        ret.scatter_(-1, tensor.unsqueeze(-1), 1)
        return ret


def switch_gating(logits, capacity=-1, need_l_aux=False, dispatchV2=False):
    """Implements M6-T switch gating with prototypes."""

    # NOTE(msb) softmax requires FP32
    gates = F.softmax(logits, dim=-1, dtype=torch.float)
    gates = gates.to(logits.dtype)

    # gates has shape of szf
    num_tokens = gates.shape[0]
    num_experts = gates.shape[2]
    # capacity = 1.25s/f
    capacity = math.ceil(1.25 * num_tokens / num_experts) if capacity < 0 else capacity
    # if torch.distributed.get_rank() == 0:
    #     logger.info(">>> gating capacity: {}".format(capacity))

    # assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=-1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Compute l_aux
    if need_l_aux:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.mean(me * ce)
    else:
        l_aux = torch.zeros(1).mean()

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=-1)

    if dispatchV2:
        # Note(zy): get rid of 'szfc' which is almost all zero,
        # 'sz*2' [f_idx, c_idx] instead
        weight = torch.sum(gates * mask1, dim=-1)  # sz
        dispatch_idx = torch.stack([indices1_s * torch.sum(mask1, dim=-1), locations1_s], dim=-1)  # sz2, idx from 0
        return (
            l_aux.to(logits.dtype),
            weight.to(logits.dtype),
            (dispatch_idx, num_experts, capacity),
        )
    else:
        # Calculate combine_weights and dispatch_mask
        gates1 = gates * mask1  # einsum("sze,sze->sze")
        locations1_sc = one_hot(locations1_s, num_classes=capacity)
        combine1_sec = gates1.unsqueeze(3) * locations1_sc.unsqueeze(2)  # einsum("sze,szc->szec")
        combine_weights = combine1_sec
        dispatch_mask = combine_weights.bool()

        return (
            l_aux.to(logits.dtype),
            combine_weights.to(logits.dtype),
            dispatch_mask,
        )


def switch_gating_outer_batch(logits, bs, capacity=-1, need_l_aux=False, dispatchV2=False):
    """
    Implements M6-T switch gating with prototypes.
    note(zy): (sb)ze ###-> bsze -> bszec
              bring batch dim out of gate compute
    """

    # (sb)zf -> sbzf
    logits = logits.reshape(-1, bs, *logits.shape[1:])

    # NOTE(msb) softmax requires FP32
    gates = F.softmax(logits, dim=-1, dtype=torch.float)
    gates = gates.to(logits.dtype)

    # gates has shape of sbzf
    num_tokens = gates.shape[0]
    num_experts = gates.shape[3]
    # capacity = 1.25s/f
    capacity = math.ceil(1.25 * num_tokens / num_experts) if capacity < 0 else capacity
    # if torch.distributed.get_rank() == 0:
    #     logger.info(">>> gating capacity: {}".format(capacity))

    # assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=-1)  # sbz
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Compute l_aux
    if need_l_aux:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.mean(me * ce)
    else:
        l_aux = torch.zeros(1).mean()

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=-1)

    if dispatchV2:
        weight = torch.sum(gates * mask1, dim=-1)  # sbz
        dispatch_idx = torch.stack([indices1_s * torch.sum(mask1, dim=-1), locations1_s], dim=-1)  # sbz2, idx from 0
        return (
            l_aux.to(logits.dtype),
            weight.to(logits.dtype),
            (dispatch_idx, num_experts, capacity),
        )
    else:
        # Calculate combine_weights and dispatch_mask
        gates1 = gates * mask1  # einsum("sbzf,sbzf->sbzf")
        locations1_sc = one_hot(locations1_s, num_classes=capacity)  # sbzc
        combine1_sec = gates1.unsqueeze(4) * locations1_sc.unsqueeze(3)  # einsum("sbzf,sbzc->sbzfc")
        combine_weights = combine1_sec
        dispatch_mask = combine_weights.bool()
        return (
            l_aux.to(logits.dtype),
            combine_weights.to(logits.dtype),
            dispatch_mask,
        )


class SwitchGate(torch.nn.Module):
    """Gate module which implements SwitchGate
       with prototypes as described in M6-T.
    ::
        gate = SwitchGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
        capacity (int, optional):
            positive if custom capacity (default: -1)
        need_l_aux (bool, optional):
            whether need l_aux
    """

    def __init__(self, model_dim, num_experts, capacity=-1, need_l_aux=False):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.capacity = capacity
        self.need_l_aux = need_l_aux

    def forward(self, input, bs, prototypes, outer_batch=False, dispatchV2=False):
        # sm -> se -> szf
        logits = self.wg(input)
        logits = logits.reshape(input.shape[0], prototypes, -1)
        if outer_batch:
            return switch_gating_outer_batch(
                logits,
                bs,
                capacity=self.capacity,
                need_l_aux=self.need_l_aux,
                dispatchV2=dispatchV2,
            )
        else:
            return switch_gating(
                logits,
                capacity=self.capacity,
                need_l_aux=self.need_l_aux,
                dispatchV2=dispatchV2,
            )
