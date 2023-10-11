import math
import operator
from typing import Callable, Dict

import torch

_DEVICE = "meta"


def torch_nn_embedding(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device=_DEVICE)


def torch_nn_functional_embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return torch.empty(*input.shape, weight.shape[-1], device=_DEVICE)


def torch_nn_layernorm(self, input):
    return input


def torch_nn_groupnorm(self, input):
    return input


def torch_nn_linear(self, input):
    return torch.empty(input.shape[:-1] + (self.out_features,), device=_DEVICE)


def torch_relu(x):
    return x


def torch_nn_relu(self, x):
    return x


def torch_nn_functional_relu(x, inplace=False):
    if not inplace:
        raise ValueError("Don't support in-place functional.relu for MetaTensor analysis")
    return x


def torch_where(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device=_DEVICE) + x.to(device=_DEVICE) + y.to(device=_DEVICE)


def torch_abs(input, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    return input


def torch_arange(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        start = int(end)
    if isinstance(step, float):
        step = int(step)
    step = kwargs.get("step", step)
    dtype = kwargs.get("dtype")
    return torch.empty((end - start) // step, dtype=dtype, device=_DEVICE)


def torch_cat(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + dim
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    dim_1 = dim + 1
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim_1:]
    return torch.empty(final_shape, device=_DEVICE)


def torch_stack(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return torch.empty(shape, device=_DEVICE)


def torch_add(input, other, *, alpha=1, out=None):
    if not isinstance(input, torch.Tensor):
        return torch.empty_like(other, device=_DEVICE)
    if not isinstance(other, torch.Tensor):
        return torch.empty_like(input, device=_DEVICE)
    max_length = max(input.dim(), other.dim())
    input_shape = list(input.shape) + [1] * (max_length - input.dim())
    other_shape = list(other.shape) + [1] * (max_length - other.dim())
    shape = []
    for i in range(max_length):
        shape.append(max(input_shape[i], other_shape[i]))
    return torch.empty(shape, device=_DEVICE)


def torch_mul(input, other, *, out=None):
    return torch_add(input, other, out=out)


def torch_tensor_mul(self, other):
    return torch_mul(self, other)


def torch_matmul(input, other, *, out=None):
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:
        shape = (input.size(0),)
    else:
        max_length = max(input.dim(), other.dim())
        shape1 = list(input.shape)
        shape2 = list(other.shape)
        if d1 == 1:
            shape1 = [1] + shape1
        if d2 == 1:
            shape2.append(1)
        shape1 = [-1] * (max_length - d1) + list(input.shape)
        shape2 = [-1] * (max_length - d2) + list(other.shape)
        shape = []
        for i in range(max_length):
            shape.append(max(shape1[i], shape2[i]))
        shape[-2] = shape1[-2]
        shape[-1] = shape2[-1]
        if d1 == 1:
            shape.pop(-2)
        if d2 == 1:
            shape.pop(-1)
    if shape is None:
        return torch.tensor(0.0, device=_DEVICE)
    return torch.empty(*shape, device=_DEVICE)


def torch_bmm(input, mat2, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    batch_size, n, m = input.shape
    _, _, p = mat2.shape
    return torch.empty(batch_size, n, p, device=_DEVICE)


def torch_einsum(equation, *operands):
    # TODO: infer shape without performing the computation, this might be quite hard.
    concrete_operands = (torch.empty_like(operand, device=_DEVICE) for operand in operands)
    return torch.einsum(equation, *concrete_operands).to(_DEVICE)


def torch_tensor_repeat(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device=_DEVICE)


def torch_index_select(input, dim, index, *, out=None):
    shape = list(input.shape)
    shape[dim] = len(index)
    return torch.empty(*shape, device=_DEVICE)


def torch_tensor_index_select(self, dim, index):
    return torch_index_select(self, dim, index)


def torch_roll(input, shifts, dims=None):
    return input


def torch_flip(input, dims):
    return input


def torch_tensor_flip(self, dims):
    return self


def torch_nn_conv1d(self, input):
    l_in = input.shape[-1]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        l_out = math.floor(
            (l_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        shape[-1] = l_out
    shape[-2] = self.out_channels
    return torch.empty(shape, device=_DEVICE)


def torch_nn_conv2d(self, input):
    h_in, w_in = input.shape[-2:]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor(
            (h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        w_out = math.floor(
            (w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        shape[-2:] = [h_out, w_out]
    shape[-3] = self.out_channels
    return torch.empty(shape, device=_DEVICE)


def torch_squeeze(input, dim=None):
    shape = list(input.shape)
    if dim is not None:
        if dim < 0:
            dim = input.dim() + dim
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        new_shape = []
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    return torch.empty(shape, device=_DEVICE)


def torch_tensor_squeeze(self, dim=None):
    return torch_squeeze(self, dim)


def torch_unsqueeze(input, dim):
    shape = list(input.shape)
    if dim < 0:
        dim = input.dim() + 1 + dim
    shape.insert(dim, 1)
    return torch.empty(shape, device=_DEVICE)


def torch_tensor_unsqueeze(self, dim):
    return torch_unsqueeze(self, dim)


def torch_unique_consecutive(input, **kwargs):
    output = torch.unique_consecutive(torch.zeros_like(input, device=_DEVICE), **kwargs)
    if isinstance(output, torch.Tensor):
        return output.to(_DEVICE)
    else:
        return tuple(map(output, lambda x: x.to(_DEVICE)))


def torch_nn_functional_one_hot(tensor, num_classes=-1):
    if num_classes < 0:
        raise ValueError("Don't support automatic num_classes inference for MetaTensor analysis")
    shape = list(tensor.shape) + [num_classes]
    return torch.empty(shape, device=_DEVICE)


def torch_nn_mseloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device=_DEVICE)


def torch_nn_crossentropyloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device=_DEVICE)


def torch_nn_bcewithlogitsloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device=_DEVICE)


def attention_func(self, hidden_states, *args, **kwargs):
    shape = hidden_states.shape
    return torch.empty(shape, device=_DEVICE)


def mlp_func(self, hidden_states, *args, **kwargs):
    shape = hidden_states.shape
    return torch.empty(shape, device=_DEVICE)


def transformer_block_func(self, hidden_states, *args, **kwargs):
    shape = hidden_states.shape
    return torch.empty(shape, device=_DEVICE)


def operator_getitem(a, b):
    def to_concrete(t):
        if isinstance(t, torch.Tensor):
            concrete = torch.ones_like(t, device=_DEVICE)
            if concrete.dtype in [
                torch.float16,
                torch.float32,
                torch.float64,
                torch.int32,
            ]:
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    if isinstance(a, torch.Tensor):
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            b = to_concrete(b)
        return operator.getitem(torch.empty_like(a, device=_DEVICE), b).to(_DEVICE)
    return operator.getitem(a, b)


_MANUAL_META_OVERRIDES: Dict[Callable, Callable] = {
    torch.nn.Embedding: torch_nn_embedding,
    torch.nn.functional.embedding: torch_nn_functional_embedding,
    torch.nn.LayerNorm: torch_nn_layernorm,
    torch.nn.GroupNorm: torch_nn_groupnorm,
    torch.nn.Linear: torch_nn_linear,
    torch.relu: torch_relu,
    torch.nn.functional.relu: torch_nn_functional_relu,
    torch.nn.ReLU: torch_nn_relu,
    torch.where: torch_where,
    torch.abs: torch_abs,
    torch.arange: torch_arange,
    torch.cat: torch_cat,
    torch.stack: torch_stack,
    torch.add: torch_add,
    torch.mul: torch_mul,
    torch.Tensor.mul: torch_tensor_mul,
    torch.matmul: torch_matmul,
    torch.bmm: torch_bmm,
    torch.einsum: torch_einsum,
    torch.Tensor.repeat: torch_tensor_repeat,
    torch.roll: torch_roll,
    torch.flip: torch_flip,
    torch.Tensor.flip: torch_tensor_flip,
    torch.index_select: torch_index_select,
    torch.Tensor.index_select: torch_tensor_index_select,
    torch.nn.Conv1d: torch_nn_conv1d,
    torch.nn.Conv2d: torch_nn_conv2d,
    torch.squeeze: torch_squeeze,
    torch.Tensor.squeeze: torch_tensor_squeeze,
    torch.unsqueeze: torch_unsqueeze,
    torch.Tensor.unsqueeze: torch_tensor_unsqueeze,
    torch.unique_consecutive: torch_unique_consecutive,
    torch.nn.functional.one_hot: torch_nn_functional_one_hot,
    torch.nn.MSELoss: torch_nn_mseloss,
    torch.nn.CrossEntropyLoss: torch_nn_crossentropyloss,
    torch.nn.BCEWithLogitsLoss: torch_nn_bcewithlogitsloss,
}


def register_meta_overrides(orig_target, meta_target):
    _MANUAL_META_OVERRIDES[orig_target] = meta_target
