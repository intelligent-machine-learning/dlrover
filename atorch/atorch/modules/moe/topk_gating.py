import math

import torch
import torch.nn.functional as F

from .switch_gating import one_hot

# from atorch.common.log_utils import default_logger as logger


def topk_gating(logits, k=2, capacity=-1, need_l_aux=False, dispatchV2=False):
    """Implements Top2Gating with prototypes on logits."""

    # NOTE(msb) softmax requires FP32
    gates = F.softmax(logits, dim=-1, dtype=torch.float)
    gates = gates.to(logits.dtype)

    # gates has shape of SE
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

    # Calculate combine_weights and dispatch_mask
    gates1_s = (gates * mask1).sum(dim=-1)  # einsum("sze,sze->sz")
    denom_s = gates1_s

    gates_s_list = [gates1_s]
    mask_list = [mask1]
    locations_s_list = [locations1_s]
    indices_s_list = [indices1_s * torch.sum(mask1, dim=-1)]  # expert idx
    last_remain_logits = logits
    last_mask = mask1
    previous_masks = mask1
    combine_weights = 0.0

    for _ in range(2, k + 1):
        last_remain_logits = last_remain_logits.masked_fill(last_mask.bool(), float("-inf"))
        indices_s = torch.argmax(last_remain_logits, dim=-1)
        mask = one_hot(indices_s, num_classes=num_experts)
        locations = torch.cumsum(mask, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations = locations + torch.sum(previous_masks, dim=0, keepdim=True)
        # Remove locations outside capacity from mask
        mask *= torch.lt(locations, capacity)
        last_mask = mask
        previous_masks = previous_masks + mask
        mask_list.append(mask)
        # Store the capacity location for each token
        locations_s = torch.sum(locations * mask, dim=-1)
        locations_s_list.append(locations_s)
        indices_s_list.append(indices_s * torch.sum(mask, dim=-1))  # expert idx
        # Normalize gate probabilities
        gates_s = (gates * mask).sum(dim=-1)  # einsum("sze,sze->sz")
        denom_s = denom_s + gates_s
        gates_s_list.append(gates_s)

    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)

    if dispatchV2:
        weight_lst, dispatch_idx_lst = [], []
        for gates_s, indices_s, locations_s in zip(gates_s_list, indices_s_list, locations_s_list):
            gates_s = gates_s / denom_s
            weight_lst.append(gates_s.to(logits.dtype))  # [sz * k]
            dispatch_idx_lst.append(torch.stack([indices_s, locations_s], dim=-1))  # [sz2 * k], idx from 0
        weight = torch.cat(weight_lst, dim=0)  # (ks)z
        dispatch_idx = torch.cat(dispatch_idx_lst, dim=0)  # (ks)z2, idx from 0
        return (
            l_aux.to(logits.dtype),
            weight,
            (dispatch_idx, num_experts, capacity),
        )
    else:
        for gates_s, mask, locations_s in zip(gates_s_list, mask_list, locations_s_list):
            gates_s = gates_s / denom_s
            # Calculate combine_weights and dispatch_mask
            gates = gates_s.unsqueeze(-1) * mask  # einsum("sz,sze->sze")
            locations_sc = one_hot(locations_s, num_classes=capacity)
            combine_sec = gates.unsqueeze(3) * locations_sc.unsqueeze(2)  # einsum("sze,szc->szec")
            combine_weights = combine_weights + combine_sec
        dispatch_mask = combine_weights.bool()
        return (
            l_aux.to(logits.dtype),
            combine_weights.to(logits.dtype),
            dispatch_mask,
        )


class TopkGate(torch.nn.Module):
    """Gate module which implements TopkGating as described in Gshard_.
       with prototypes as described in M6-T.
    ::
        gate = SwitchGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
        k (int, optional):
            number of top experts that tokens are dispatched to
        capacity (int, optional):
            positive if custom capacity (default: -1)
        need_l_aux (bool, optional):
            whether need l_aux
    """

    def __init__(self, model_dim, num_experts, k=2, capacity=-1, need_l_aux=False):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = k
        self.capacity = capacity
        self.need_l_aux = need_l_aux

    def forward(self, input, bs, prototypes, outer_batch=False, dispatchV2=False):
        # sm -> se -> szf
        logits = self.wg(input)
        logits = logits.reshape(input.shape[0], prototypes, -1)
        if outer_batch:
            raise NotImplementedError
        else:
            return topk_gating(
                logits,
                k=self.k,
                capacity=self.capacity,
                need_l_aux=self.need_l_aux,
                dispatchV2=dispatchV2,
            )
