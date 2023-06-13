import torch
import torch.distributed as dist
from torch import autograd


class AllGatherQMicro(autograd.Function):
    @staticmethod
    def forward(ctx, q_micro, group=None):
        """
        input:
            q_micro (tensor of [BatchDim, MultiHeadDim, qMicroDim, HiddenDim]):
                local micro q
            group (ProcessGroup, optional):
                The sequence parallel process group to work on.
        return:
            q (tensor of [BatchDim, MultiHeadDim, qDim, HiddenDim]):
                all-gathered q emb
        """
        seq_world_size = group.size() if group else dist.get_world_size()

        q_micro_list = [torch.zeros_like(q_micro) for _ in range(seq_world_size)]
        dist.all_gather(q_micro_list, q_micro, group=group)

        # cat at qDim
        q = torch.cat(q_micro_list, dim=2)
        ctx.group = group
        return q

    @staticmethod
    def backward(ctx, q_grad):
        """
        input:
            q_grad (tensor of [BatchDim, MultiHeadDim, qDim, HiddenDim]):
                all-gathered q's grad
        return:
            q_micro_grad
            (tensor of [BatchDim, MultiHeadDim, qMicroDim, HiddenDim]):
                local micro q's grad
        """
        seq_world_size = ctx.group.size() if ctx.group else dist.get_world_size()
        q_grad_list = [q_micro_grad.contiguous() for q_micro_grad in q_grad.chunk(seq_world_size, dim=2)]
        q_micro_grad = torch.zeros_like(q_grad_list[0])
        dist.reduce_scatter(q_micro_grad, q_grad_list, group=ctx.group)
        return q_micro_grad, None


class ReduceScatterContext(autograd.Function):
    @staticmethod
    def forward(ctx, c, group=None):
        """
        input:
            c (tensor of [BatchDim, MultiHeadDim, qDim, HiddenDim]):
                context, from matmul(local attn map, local v)
            group (ProcessGroup, optional):
                The sequence parallel process group to work on.
        return:
            c_micro (tensor of [BatchDim, MultiHeadDim, qMicroDim, HiddenDim]):
                local micro c
        """
        seq_world_size = group.size() if group else dist.get_world_size()
        c_micro_list = [c_micro.contiguous() for c_micro in c.chunk(seq_world_size, dim=2)]
        c_micro = torch.zeros_like(c_micro_list[0])
        dist.reduce_scatter(c_micro, c_micro_list, group=group)
        ctx.group = group
        return c_micro

    @staticmethod
    def backward(ctx, c_micro_grad):
        """
        input:
            c_micro_grad (tensor of [BatchDim, MultiHeadDim, qMicroDim, HiddenDim]):
                local micro c's grad
        return:
            c_grad
            (tensor of [BatchDim, MultiHeadDim, qDim, HiddenDim]):
                context's grad
        """
        c_micro_grad = c_micro_grad.contiguous()
        seq_world_size = ctx.group.size() if ctx.group else dist.get_world_size()

        c_micro_grad_list = [torch.zeros_like(c_micro_grad) for _ in range(seq_world_size)]
        dist.all_gather(c_micro_grad_list, c_micro_grad, group=ctx.group)

        # cat at qDim
        c_grad = torch.cat(c_micro_grad_list, dim=2)
        return c_grad, None
