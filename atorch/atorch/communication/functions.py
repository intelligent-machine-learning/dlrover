"""
Adapted from Megatron-LM,
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py
"""


import torch

from atorch.distributed.distributed import parallel_group


def _gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = parallel_group("expert")
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _gather_along_first_dim_moe(input_, async_op=False, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    group = parallel_group("expert")
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    if use_global_buffer:
        raise NotImplementedError()
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

    if async_op:
        handle = torch.distributed._all_gather_base(output, input_.contiguous(), group=group, async_op=async_op)
        return output, handle

    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_, async_op=False, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group."""
    group = parallel_group("expert")
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size

    if use_global_buffer:
        raise NotImplementedError()
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

    if async_op:
        handle = torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group, async_op=async_op)
        return output, handle

    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, async_op=False):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )

        if async_op:
            a2a_handle = torch.distributed.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=async_op,
            )
            return output, a2a_handle

        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output, grad_handle=None):
        return (
            None,
            _AllToAll.apply(ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
            None,
        )


class _GatherFromExpertParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from expert parallel region and concatenate."""  # TODO

    @staticmethod
    def forward(ctx, input_, async_op=False, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim_moe(input_, async_op, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output, grad_handle=None):
        use_global_buffer = ctx.use_global_buffer
        return _reduce_scatter_along_first_dim_moe(grad_output, use_global_buffer=use_global_buffer), None, None


class _ReduceScatterToExpertParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, async_op=False, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim_moe(input_, async_op, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output, grad_handle=None):
        use_global_buffer = ctx.use_global_buffer
        return _gather_along_first_dim_moe(grad_output, use_global_buffer), None, None


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None, async_op=False):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_, async_op)


def gather_from_expert_parallel_region_to_moe(input_, async_op=False, use_global_buffer=False):
    return _GatherFromExpertParallelRegionToMOE.apply(input_, async_op, use_global_buffer)


def reduce_scatter_to_expert_parallel_region_from_moe(input_, async_op=False, use_global_buffer=False):
    return _ReduceScatterToExpertParallelRegionFromMOE.apply(input_, async_op, use_global_buffer)
