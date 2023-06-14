import numpy as np
import torch
import torch.distributed as dist
from torch.fx.passes.shape_prop import TensorMetadata


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator


def rank_list_is_sorted(shuffled_list):
    """check if a list is shuffled

    Args:
        shuffled_list: a list of ranks

    Returns:
        True if the list is sorted, False otherwise
    """
    return all(shuffled_list[i] < shuffled_list[i + 1] for i in range(len(shuffled_list) - 1))


def split_tensor_along_shard_dim(tensor, shard_dim, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along shard_dim

    Args:
        tensor: input tensor.
        shard_dim: along which dimension to shard the tensor
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of split tensors
    """
    shard_dim_size = divide(list(tensor.size())[shard_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, shard_dim_size, dim=shard_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def rearrange_tensor_list(tensor_list, ordered_ranks, ranks):
    """Assumes tensor_list is ordered according to ordered_ranks, rearrange
    tensor_list by the order of ranks

    Args:
        tensor_list (list): a list of tensors to be rearranged
        ordered_ranks (list): a list of ranks by which the tensor_list
            is ordered
        ranks (list): a list of ranks by which the tensor_list is to be reordered

    Returns:
        a list of properly rearranged tensors
    """
    tensor_list = [tensor_list[ordered_ranks.index(rank)] for rank in ranks]
    return tensor_list


def generate_output_tensor_list(input_, cur_shard):
    """Generate a list of empty tensors to receive sharded input_
    from other ranks.

    Args:
        input_ (torch.Tensor): the input_ tensor to be sharded and broadcast among
            the process_group
        cur_shard (ShardingSpec): the sharding spec of the sharded input_

    Returns:
        a list of empty tensors of properly shard size
    """
    world_size = dist.get_world_size(cur_shard.group)
    output_size = list(input_.size())
    output_size[-1] = output_size[-1] // world_size
    output_tensor_list = [torch.empty(output_size, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]
    return output_tensor_list


def _compute_tensor_size(tensor_shape):
    tensor_size = 0

    def _aggregate_memory(t_shape):
        nonlocal tensor_size
        if isinstance(t_shape, TensorMetadata):
            activation_dtype = t_shape.dtype
            activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
            # cost of activation
            activation_size = np.prod(list(t_shape.shape)) * activation_size_per_elem_bytes
            tensor_size += activation_size
        elif isinstance(t_shape, (list, tuple)):
            for t_s in t_shape:
                _aggregate_memory(t_s)
        elif isinstance(t_shape, dict):
            for t_s in t_shape.values():
                _aggregate_memory(t_s)
        else:
            return

    _aggregate_memory(tensor_shape)
    return tensor_size


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indecies in [fist, last)

    Adapted from Megatron-LM https://github.com/NVIDIA/Megatron-LM
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, ranks):
        rank_index = ranks.index(rank)
        index_f = rank_index * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, ranks):
        world_size = len(ranks)
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, ranks)


"""The following codes are also taken from Megatron-LM,
in support of activation checkpointing for Tensor Parallel models.
"""


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor
