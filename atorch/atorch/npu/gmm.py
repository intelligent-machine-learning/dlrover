import torch
from torch.cuda.amp.autocast_mode import _cast, autocast
from torch.library import impl

from .op_builder.gmm_builder import GMMOpBuilder
from .op_builder.npu_builder import ATORCH_NPU_LIBRARY


class GMMFunction(torch.autograd.Function):
    npu_ops = GMMOpBuilder().load()

    @staticmethod
    def forward(ctx, x, weight, bias, group_list, group_type):
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and not (
            isinstance(group_list, list) and all(isinstance(x, int) for x in group_list)
        ):
            raise TypeError(
                f"group_list must be a List of int to compute gradients of x and weight, got {type(group_list)}.!"
            )
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        outputs = GMMFunction.npu_ops.npu_gmm([x], [weight], bias, group_list, group_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        dx, dw, dbias = GMMFunction.npu_ops.npu_gmm_backward([grad_outputs], [x], [weight], ctx.group_list)
        dbias = None if len(dbias) == 0 else dbias[0]
        # for values with 0 in GMM tensorlist, they cannot be processed on the kernel side currently,
        # and the result should be reset to zero
        for i, _ in enumerate(ctx.group_list):
            if i == 0:
                if ctx.group_list[0] == 0:
                    dw[0][0].fill_(0)
                continue
            if ctx.group_list[i] == ctx.group_list[i - 1]:
                dw[0][i].fill_(0)

        return dx[0], dw[0], dbias, None, None


@impl(ATORCH_NPU_LIBRARY, "npu_gmm.List", "PrivateUse1")
@impl(ATORCH_NPU_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
def _npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg0 must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"arg1 must be a torch.Tensor, got {type(weight)}.")
    if not isinstance(bias, (torch.Tensor, type(None))):
        raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}.")
    if not (
        isinstance(group_list, (torch.Tensor, type(None)))
        or (isinstance(group_list, list) and all(isinstance(x, int) for x in group_list))
    ):
        raise TypeError(f"group_list must be a List of int, torch.Tensor or None, got {type(group_list)}.")
    if isinstance(group_list, torch.Tensor):
        if len(group_list.shape) > 1:
            raise ValueError(
                f"If group_list is not None, it must be an one-dimensional tensor, "
                f"got dimension of group_list: {len(group_list.shape)}!"
            )
        if group_list.dtype != torch.int64:
            raise TypeError(
                f"group_list must be a List of int, got group_list type: {type(group_list)}, "
                f"dtype: {group_list.dtype}!"
            )
    if not isinstance(group_type, (int, type(None))):
        raise TypeError(f"group_type must be an int or None, got {type(group_type)}.")
    # Ensure all tensors on the same device
    x_device = x.device
    device_warning = "Expected all tensors to be on the same device, but found at least two devices"
    if weight.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {weight.device}(arg1)!")
    if bias is not None and bias.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {bias.device}(bias)!")
    if isinstance(group_list, torch.Tensor) and group_list.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {group_list.device}(group_list)!")

    return GMMFunction.apply(x, weight, bias, group_list, group_type)


def npu_gmm(x, weight, batch_sizes, trans_b):
    weight = weight.transpose(1, 2) if trans_b else weight
    # size of groupList can not be 1.If expected group num is 1, groupList should be nullptr.
    if len(batch_sizes) == 1:
        return x @ weight
    from itertools import accumulate

    group_list = list(accumulate(batch_sizes.tolist()))
    group_type = 0
    if torch.is_autocast_enabled():
        cur_dtype = torch.get_autocast_gpu_dtype()
        with autocast(enabled=False):
            x = _cast(x, cur_dtype)
            weight = _cast(weight, cur_dtype)
    return torch.ops.atorch_npu.npu_gmm(x, weight, bias=None, group_list=group_list, group_type=group_type)
