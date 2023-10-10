import torch
import torch.distributed as dist
import torch.nn.functional as F


def _all_gather_multishape_tensor(tensor, padding_size, all_real_tensor_size, group, world_size):
    """
    All gather tensors with different shape
    Args:
        tensor: local dense tensor to be all_gathered
        padding_size: padding local tensor to max global size
        all_real_tensor_size: real tensor size in all process group
        group: process group for all_gather
        world_size: process group's total size
    """
    device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"

    # prepare input for all_gather
    padded_tensor = F.pad(tensor, [0, padding_size])

    # setup placement for all_gather
    result_placement = [torch.zeros_like(padded_tensor).to(device) for _ in range(world_size)]
    dist.all_gather(result_placement, padded_tensor, group=group)

    # remove padding
    for i, t in enumerate(result_placement):
        current_padding_size = result_placement[i].shape[-1] != all_real_tensor_size[i]
        if current_padding_size:
            result_placement[i] = t.index_select(-1, torch.tensor(range(all_real_tensor_size[i])).to(device))
    return result_placement


def all_reduce_sparse(sparse_tensor, group=None):
    """
    Args:
        sparse_tensor: local sparse tensor to be all_reduced
        group: process group for all_reduce
    """
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    world_size = dist.get_world_size(group)
    device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"

    size = sparse_tensor.size()
    local_nnz, local_indices, local_values = sparse_tensor._nnz(), sparse_tensor._indices(), sparse_tensor._values()
    # setup nnz placement
    nnz_placement = [torch.tensor(0).to(device) for _ in range(world_size)]
    # all gather nnz
    dist.all_gather(nnz_placement, torch.tensor(local_nnz).to(device))

    # get padding size
    max_nnz = max(nnz_placement)
    padding_size = max_nnz - local_nnz
    indices = _all_gather_multishape_tensor(local_indices, padding_size, nnz_placement, group, world_size)
    values = _all_gather_multishape_tensor(local_values, padding_size, nnz_placement, group, world_size)
    all_gathered_sparse_tensors = [torch.sparse_coo_tensor(indices[i], values[i], size) for i in range(world_size)]

    res = all_gathered_sparse_tensors[0]
    for i in range(1, len(all_gathered_sparse_tensors)):
        res = res + all_gathered_sparse_tensors[i]
    return res.coalesce()
