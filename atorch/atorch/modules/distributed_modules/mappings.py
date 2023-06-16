# Implements autograd-collective communications needed for distributed modules/graph transform

import torch
import torch.distributed as dist

import atorch
from atorch.distributed.distributed import parallel_group, parallel_group_and_ranks, parallel_group_size, parallel_rank
from atorch.modules.distributed_modules.utils import (
    divide,
    generate_output_tensor_list,
    rank_list_is_sorted,
    rearrange_tensor_list,
    split_tensor_along_shard_dim,
)
from atorch.utils.version import torch_version

if torch_version() <= (1, 12, 1):
    from torch.distributed import _all_gather_base as torch_all_gather_base
else:
    from torch.distributed import all_gather_into_tensor as torch_all_gather_base


def _reduce(input_, group=None):
    """ "All reduce the input_ tensor across group.

    Args:
        input_: the input_ tensor to be all-reduced.
        group: the process group on which to perform all_reduc. Defaults to group 'model',
            can be the group name.

    Returns:
        reduced input
    """
    if group is None:
        group = "tensor"
    if isinstance(group, str):
        group = parallel_group(group)
    # All reduce
    torch.distributed.all_reduce(input_, group=group)

    return input_


# FIXME only support 1d reshuffle
def _reshuffle(input_, prev_shard, cur_shard):
    """Reshuffles the local input_ among the process group according to
    prev_shard.ranks and cur_shard.ranks.

    Args:
        input_: the tensor held locally by this rank.
        prev_shard: the sharding spec of the local tensor.
        cur_shard: the target sharding spec the local tensor should be.

    Returns:
        The local tensor received from the target rank specified in cur_shard.
    """
    process_group = prev_shard.group
    sorted_ranks = sorted(prev_shard.ranks)
    world_size = dist.get_world_size(process_group)
    # use the rank in default group, as ranks contains ranks in default group
    current_rank = dist.get_rank()
    shard_size = input_.size(prev_shard.dims[0])
    input_split_sizes = [0] * world_size
    output_split_sizes = [0] * world_size

    # send to new_rank
    idx = prev_shard.ranks.index(current_rank)
    new_rank = cur_shard.ranks[idx]
    new_group_idx = sorted_ranks.index(new_rank)
    input_split_sizes[new_group_idx] = shard_size
    # receive from rec_rank
    rec_idx = cur_shard.ranks.index(current_rank)
    rec_rank = prev_shard.ranks[rec_idx]
    rec_group_idx = sorted_ranks.index(rec_rank)
    output_split_sizes[rec_group_idx] = shard_size

    input_ = input_.transpose(0, prev_shard.dims[0]).contiguous()
    gathered_input = torch.empty(input_.size(), device=input_.device, dtype=input_.dtype)
    # all2all
    input_ = dist.nn.functional.all_to_all_single(
        gathered_input,
        input_,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=process_group,
    )
    return input_.transpose(0, prev_shard.dims[0]).contiguous()


# FIXME only support 1d reshard
def _reshard(input_, prev_shard, cur_shard):
    """Reshard the input_ tensor on dimensions specified in cur_shard.

    Args:
        input_: the tensor held locally by this rank.
        prev_shard: the sharding spec of the local tensor.
        cur_shard: the target sharding spec the local tensor should be.

    Returns:
        the local tensor input_ sharded on dimensions specified in cur_shard.
    """
    world_size = dist.get_world_size(prev_shard.group)
    input_tensor_list = split_tensor_along_shard_dim(input_, cur_shard.dims[0], world_size, True)
    if rank_list_is_sorted(cur_shard.ranks):
        input_tensor_list = rearrange_tensor_list(input_tensor_list, sorted(cur_shard.ranks), cur_shard.ranks)

    output_tensor_list = generate_output_tensor_list(input_, cur_shard)
    output_tensor_list = dist.nn.functional.all_to_all(output_tensor_list, input_tensor_list, group=cur_shard.group)
    if rank_list_is_sorted(prev_shard.ranks):
        output_tensor_list = rearrange_tensor_list(output_tensor_list, prev_shard.ranks, sorted(cur_shard.ranks))

    return torch.cat(output_tensor_list, dim=prev_shard.dims[0])


def split_tensor_into_1d_equal_chunks(tensor, group="tensor", new_buffer=False):
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Returns a Tensor or View with this rank's portion of the data.

    Arguments:
        tensor: The tensor to split

    Keyword Arguments:
        group (str): The parallel group on which to split the tensor
        new_buffer (bool): If True, returns a new Tensor.
                           If False, returns a view into the existing Tensor.
                           Default is False

    """
    partition_size = divide(torch.numel(tensor), parallel_group_size(group))
    start_index = partition_size * parallel_rank(group)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size, dtype=tensor.dtype, device=f"cuda:{atorch.local_rank()}", requires_grad=False
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor, group="tensor"):
    """Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
    model parallel ranks.

    Returns a new Tensor with the gathered data.

    Arguments:
        tensor: A Tensor or view of this rank's portion of the data.

    Keyword Arguments:
        group (str): The parallel group on which to split the tensor
    """
    numel_gathered = torch.numel(tensor) * parallel_group_size(group)
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype, device=torch.cuda.current_device(), requires_grad=False)
    torch_all_gather_base(gathered, tensor, group=parallel_group(group))
    return gathered


def _split_last_dim_gather_first_dim(input_, group=None):
    """Split the input tensor (by last dim) across model parallel group.
    Gather the first dimension
    This operator assumes that tensors are sharded in natural order
    of parallel group and performs no reshuffling

    Args:
        input_: input tensor to be split and gathered
        group: the process group on which to split and grather.
            Defaults to Model parallel group. if is a string,
            the group is retrieved with atorch parallel_group.

    return:
        split and gathered tensor.
    """
    if group is None:
        group = parallel_group("tensor")
    elif isinstance(group, str):
        group = parallel_group(group)

    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output_size = list(input_.size())
    output_size[-1] = output_size[-1] // world_size
    device = f"cuda:{atorch.local_rank()}"
    output_tensor_list = [torch.empty(output_size, dtype=input_.dtype, device=device) for _ in range(world_size)]
    # chunk_size = output_size[-1]
    # input_tensor_list = [input_[..., i * chunk_size : (i + 1) * chunk_size].contiguous() for i in range(world_size)]
    input_tensor_list = list(split_tensor_along_shard_dim(input_, -1, world_size, True))
    dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    output_tensor = torch.cat(output_tensor_list, dim=0)
    return output_tensor


def _split_first_dim_gather_last_dim(input_, group=None):
    """Split the input tensor (by first dim) across model parallel group.
    Gather the last dimension
    This operator assumes that tensors are sharded in natural order
    of parallel group and performs no reshuffling

    Args:
        input_: input tensor to be split and gathered
        group: the process group on which to split and grather.
            Defaults to Model parallel group. if is a string,
            the group is retrieved with atorch parallel_group.

    return:
        split and gathered tensor
    """
    if group is None:
        group = parallel_group("tensor")
    elif isinstance(group, str):
        group = parallel_group(group)

    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output_size = list(input_.size())
    output_size[0] = output_size[0] // world_size
    output_tensor_list = [
        torch.zeros(output_size, dtype=input_.dtype, device=torch.cuda.current_device()) for _ in range(world_size)
    ]
    input_ = input_.contiguous()
    # chunk_size = output_size[0]
    # input_tensor_list = [input_[i * chunk_size : (i + 1) * chunk_size, ...] for i in range(world_size)]
    input_tensor_list = list(split_tensor_along_shard_dim(input_, 0, world_size, True))
    dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    output_tensor = torch.cat(output_tensor_list, dim=-1)
    return output_tensor


def _split_shard_dim_with_reshuffle_check(input_, shard_dim, group=None, ranks=None):
    """split the designated dimension, check if ranks are shuffled, and takes the
    appropriate shard.

    Args:
        input_: the tensor to be sharded.
        shard_dim: the dimension on which to shard the input_.
        group: the process group on which the sharding is done. Defaults to group 'model'.
            Can be a str, in which case the group is retreived by parallel_group(group)
        ranks: the ranks corresponding to group. Can be None in case group is None or a str.
            Needed for performing reshuffle check. Ranks in this list should corresponds to ranks
            in default group (not necessarily 0-world_size/contiguous)

    Returns:
        The sharded local tensor.
    """
    if group is None:
        group = "tensor"
    if isinstance(group, str):
        group, ranks = parallel_group_and_ranks(group)
    world_size = dist.get_world_size(group)
    output_tensor_list = split_tensor_along_shard_dim(input_, shard_dim, world_size, True)
    # use the rank in default group, as ranks contains ranks in default group
    my_rank = dist.get_rank()
    rank_index = ranks.index(my_rank)
    output_tensor = output_tensor_list[rank_index]
    return output_tensor


def _gather_shard_dim_with_reshuffle_check(input_, shard_dim, group=None, ranks=None):
    """Gather along the designated dimension, check if ranks are shuffled, in case shuffled, reshuffle
    the gathered tensors then concate.

    Args:
        input_: the sharded local tensor held by this rank.
        shard_dim: the dimension on which this local tensor is sharded.
        group: The process group on which to collect other pieces of the tensor.  Defaults to group 'model'.
            Can be a str, in which case the group is retreived by parallel_group(group).
        ranks: the ranks corresponding to group. Can be None in case group is None or a str.
            Needed for performing reshuffle check. Ranks in this list should corresponds to ranks
            in default group (not necessarily 0-world_size/contiguous)

    Returns:
        The full tensor.
    """
    if group is None:
        group = "tensor"
    if isinstance(group, str):
        group, ranks = parallel_group_and_ranks(group)
    world_size = dist.get_world_size(group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    # use the rank in default group, as ranks contains ranks in default group
    my_rank = dist.get_rank()
    sorted_ranks = sorted(ranks)
    sorted_index = sorted_ranks.index(my_rank)
    input_ = input_.contiguous()
    tensor_list[sorted_index] = input_
    dist.all_gather(tensor_list, input_, group=group)
    if not rank_list_is_sorted(ranks):
        tensor_list = rearrange_tensor_list(tensor_list, ranks, sorted_ranks)
    output = torch.cat(tensor_list, dim=shard_dim).contiguous()
    return output


class _ReduceFromGroup(torch.autograd.Function):
    """All-reduce the input from the group."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_, group=None):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _CopyToGroup(torch.autograd.Function):
    """Pass the input to the group.
    This method installs an all_reduce in the backward pass and does nothing in forward pass"""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_, group=None):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None


class _SplitShardDimReshuffleCheck(torch.autograd.Function):
    """Split a designated dimension, check if ranks are shuffled, and takes
    the appropriate shard."""

    @staticmethod
    def forward(ctx, input_, shard_dim, group=None, ranks=None):
        ctx.group = group
        ctx.ranks = ranks
        ctx.shard_dim = shard_dim
        return _split_shard_dim_with_reshuffle_check(input_, shard_dim, group, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather_shard_dim_with_reshuffle_check(grad_output, ctx.shard_dim, ctx.group, ctx.ranks),
            None,
            None,
            None,
        )


class _GatherShardDimWithReshuffleCheck(torch.autograd.Function):
    """Gather the designated dimension, check if ranks are shuffled and reorder if necessary"""

    @staticmethod
    def forward(ctx, input_, shard_dim, group=None, ranks=None):
        ctx.group = group
        ctx.ranks = ranks
        ctx.shard_dim = shard_dim
        return _gather_shard_dim_with_reshuffle_check(input_, shard_dim, group, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _split_shard_dim_with_reshuffle_check(grad_output, ctx.shard_dim, ctx.group, ctx.ranks),
            None,
            None,
            None,
        )


class _SplitLastDimGatherFirstDim(torch.autograd.Function):
    """Split the last dimension and gather the first dimension
    Do not check if ranks is shuffled.
    """

    @staticmethod
    def forward(ctx, input_, group=None):
        ctx.group = group
        return _split_last_dim_gather_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_first_dim_gather_last_dim(grad_output, group=ctx.group), None


# -----------------
# Helper functions.
# -----------------


def copy_to_group(input_, group=None):
    return _CopyToGroup.apply(input_, group)


def reduce_from_group(input_, group=None):
    return _ReduceFromGroup.apply(input_, group)


def split_last_dim_gather_first_dim(input_, group=None):
    return _SplitLastDimGatherFirstDim.apply(input_, group)


def split_shard_dim_with_reshuffle_check(input_, shard_dim, group=None, ranks=None):
    return _SplitShardDimReshuffleCheck.apply(input_, shard_dim, group, ranks)


def gather_shard_dim_with_reshuffle_check(input_, shard_dim, group=None, ranks=None):
    return _GatherShardDimWithReshuffleCheck.apply(input_, shard_dim, group, ranks)


def split_first_dim_with_reshuffle_check(input_, group=None, ranks=None):
    return _SplitShardDimReshuffleCheck.apply(input_, 0, group, ranks)


def split_last_dim_with_reshuffle_check(input_, group=None, ranks=None):
    return _SplitShardDimReshuffleCheck.apply(input_, -1, group, ranks)


def collect_and_reshard(input_, prev_shard, cur_shard):
    """A general resharding operator that collect input_ over prev_spec.group,
    then aggregate and distribute over cur_shard.group.

    Currently only support 1d resharding.

    Args:
        input_: the local tensor to be resharded.
        prev_shard: the sharding_spec of the current local tensor.
        cur_shard: the sharding spec the resharded tensor should have.

    Returns:
        The correctly resharded local tensor.
    """
    if set(prev_shard.ranks) != set(cur_shard.ranks):
        raise ValueError("cannot handle inter group communication now")

    if len(prev_shard.dims) > 1 or len(cur_shard.dims) > 1:
        raise ValueError("cannot handle sharding of more than 1 dimension")

    if len(prev_shard.dims) == 0:
        shard_dim = cur_shard.dims[0]
        return split_shard_dim_with_reshuffle_check(input_, shard_dim, cur_shard.groups, cur_shard.ranks)

    if len(cur_shard.dims) == 0:
        shard_dim = prev_shard.dims[0]
        return gather_shard_dim_with_reshuffle_check(input_, shard_dim, cur_shard.group, prev_shard.ranks)

    if prev_shard.dims == cur_shard.dims:
        return _reshuffle(input_, prev_shard, cur_shard)
    else:
        return _reshard(input_, prev_shard, cur_shard)
