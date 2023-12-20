# Modifications Copyright 2023 AntGroups, Inc.

# Copyright (c) Tsinghua Statistical Artificial Intelligence & Learning Group.
# SPDX-License-Identifier: Apache-2.0

import itertools

import torch

ext_quantization = None
if torch.cuda.is_available():
    from atorch.ops.op_builder import QuantizationOptimizerBuilder  # type: ignore

    ext_quantization = QuantizationOptimizerBuilder().load()

lpmm_generator = None
FP_EXPONENT_BIS_MAP = {
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
}


def init_lpmm_generator(gpu, seed):
    global lpmm_generator
    if lpmm_generator is None:
        lpmm_generator = torch.Generator(device=gpu)
        if seed is None:
            seed = 3407
        lpmm_generator.manual_seed(seed)


def vectorwise_quant(x, **kwargs):
    """interface quantization function"""
    qx = x.detach()  # keep the reference of original tensor

    # save kwargs
    generated_metadata = {}
    generated_metadata["dtype"] = x.dtype
    generated_metadata["stride"] = x.stride()

    # Given a ill-conditioned/quantization-unfriendly tensor, how to normalize and/or avoid outlier?
    # scale/noramlize the original tensor
    qx, md = quant_scaling(qx, **kwargs)
    generated_metadata.update(md)

    # Given a tensor distributed in [-1/0, 1], how to quantize with best error?
    # quantize the normalized tensor
    quant_type = kwargs["quant_type"]
    b, signed = kwargs["b"], kwargs["signed"]
    if quant_type == "linear":
        MRQ, lo, hi = prepare_quant_boundary(b, signed)
        qx = atom_quant(qx, None, MRQ, lo, hi, round_type=kwargs["round_type"])
    elif quant_type in ["nonlinear", "power-1", "power-2", "power-3", "float-point", "nonlinear-nozero"]:
        if isinstance(kwargs["qmap"], torch.Tensor):
            qmap = kwargs["qmap"]
        else:
            qmap = kwargs["qmap"][(b, signed)][quant_type]
        qx = nonlinear_quant(qx, qmap, b, round_type=kwargs["round_type"])
    else:
        raise ValueError(f"Not support {quant_type} quant type.")

    return qx, generated_metadata


def vectorwise_dequant(qx, denormalized=True, **kwargs):
    """dequantization function"""
    x = qx.detach()

    # load kwargs
    dtype = kwargs["dtype"]
    stride = kwargs["stride"]

    # dequantize the quantized tensor to get a tensor in [-1/0, 1]
    quant_type = kwargs["quant_type"]
    b, signed = kwargs["b"], kwargs["signed"]
    if quant_type == "linear":
        MRQ, lo, hi = prepare_quant_boundary(b, signed)
        x = atom_dequant(x, None, MRQ)
    elif quant_type in ["nonlinear", "power-1", "power-2", "power-3", "float-point", "nonlinear-nozero"]:
        if isinstance(kwargs["qmap"], torch.Tensor):
            qmap = kwargs["qmap"]
        else:
            qmap = kwargs["qmap"][(b, signed)][quant_type]
        x = nonlinear_dequant(x, qmap, b, shape=kwargs["scaled_shape"], round_type=kwargs["round_type"])
    else:
        raise ValueError(f"Not support {quant_type} quant type.")

    # only for debug
    if not denormalized:
        return x

    # scale the dequantized tensor to get the original tensor
    scale_type = kwargs["scale_type"]
    max1 = kwargs["max1"]
    if scale_type in ["tensor", "dim0", "dim1"]:
        x = x.mul(max1)
    elif scale_type in ["rank1"]:
        dim = kwargs["dim"]
        if dim == 1:  # group
            x = x.mul(max1)
            shape = kwargs["shape"]
            x = recon_grouped_tensor(x, shape)
        else:
            max_dims = kwargs["max_dims"]
            st = _compute_sm3_scale_tensor(max_dims)
            x = x.mul(st)
    elif scale_type == "dim01":
        x = x.mul(max1)
        max_dim0 = kwargs["max_dim0"]
        x = x.mul(max_dim0)
    elif scale_type == "dim10":
        x = x.mul(max1)
        max_dim1 = kwargs["max_dim1"]
        x = x.mul(max_dim1)
    elif scale_type == "group":
        x = x.mul(max1)
        shape = kwargs["shape"]
        x = recon_grouped_tensor(x, shape)
    elif scale_type == "rank1-group":
        dim = kwargs["dim"]
        if dim == 1:  # group
            x = x.mul(max1)
            shape = kwargs["shape"]
            x = recon_grouped_tensor(x, shape)
        elif dim == 2:
            max0 = kwargs["max0"]
            gp0_shape = kwargs["gp0_shape"]
            st0 = recon_grouped2d_tensor(max0.expand(gp0_shape), kwargs["shape"])
            gp1_shape = kwargs["gp1_shape"]
            st1 = recon_grouped2d_tensor(max1.expand(gp1_shape), kwargs["Tshape"])
            st = torch.min(st0, st1.T)
            x = x.mul(st)
        else:  # rank1
            max_dims = kwargs["max_dims"]
            st = _compute_sm3_scale_tensor(max_dims)
            x = x.mul(st)
    elif scale_type == "id":
        pass
    else:
        raise NotImplementedError

    if x.stride() != stride:
        recon_x = torch.empty_strided(x.shape, stride, dtype=dtype, layout=torch.strided, device=x.device)
        recon_x.copy_(x)
        del x
        return recon_x
    else:
        x = x.to(dtype=dtype)
        return x


def quant_scaling(qx, **kwargs):
    scale_type = kwargs["scale_type"]
    generated_metadata = {}
    # reshape and scaling
    if scale_type == "tensor":
        max1 = torch.amax(torch.abs(qx), keepdim=True).to(torch.float32)  # (1, 1)
        generated_metadata["max1"] = max1
        qx = qx.div(max1)
    elif scale_type == "dim0":
        max1 = _max_reduce_except_dim(qx.abs(), 0)
        generated_metadata["max1"] = max1
        qx = qx.div(max1)
    elif scale_type == "dim1":
        max1 = _max_reduce_except_dim(qx.abs(), 1)
        generated_metadata["max1"] = max1
        qx = qx.div(max1)
    elif scale_type == "dim01":
        max_dim0 = _max_reduce_except_dim(qx.abs(), 0)
        qx = qx.div(max_dim0)
        max1 = _max_reduce_except_dim(qx.abs(), 1)
        generated_metadata["max_dim0"] = max_dim0
        generated_metadata["max1"] = max1
        qx = qx.div(max1)
    elif scale_type == "dim10":
        max_dim1 = _max_reduce_except_dim(qx.abs(), 1)
        qx = qx.div(max_dim1)
        max1 = _max_reduce_except_dim(qx.abs(), 0)
        generated_metadata["max_dim1"] = max_dim1
        generated_metadata["max1"] = max1
        qx = qx.div(max1)
    elif scale_type == "group":
        gp_sz = kwargs["gp_sz"]
        qx = group_tensor(qx, gp_sz)  # (num_gp, gp_sz)
        max1 = _max_reduce_except_dim(qx.abs(), 0)
        qx = qx.div(max1)
        generated_metadata["max1"] = max1
    elif scale_type == "rank1":
        generated_metadata["dim"] = qx.dim()
        if qx.dim() == 1:  # group
            gp_sz = 128
            qx = group_tensor(qx, gp_sz)  # (num_gp, gp_sz)
            max1 = _max_reduce_except_dim(qx.abs(), 0)
            qx = qx.div(max1)
            generated_metadata["max1"] = max1
        else:
            max_dims = get_sm3_statistics(qx.abs())
            st = _compute_sm3_scale_tensor(max_dims)
            generated_metadata["max_dims"] = max_dims
            generated_metadata["max1"] = None
            qx = qx.div(st)
    elif scale_type == "rank1-group":
        gp_sz = kwargs["gp_sz"]
        generated_metadata["dim"] = qx.dim()
        if qx.dim() == 1:  # group
            gp_sz = 128
            qx = group_tensor(qx, gp_sz)  # (num_gp, gp_sz)
            max1 = _max_reduce_except_dim(qx.abs(), 0)
            qx = qx.div(max1)
            generated_metadata["max1"] = max1
        elif qx.dim() == 2:
            generated_metadata["Tshape"] = qx.T.shape
            gp0_qx = group2d_tensor(qx, gp_sz)  # (num_gp, gp_sz)
            max0 = _max_reduce_except_dim(gp0_qx.abs(), 0)
            generated_metadata["max0"] = max0
            st0 = recon_grouped2d_tensor(max0.expand_as(gp0_qx), qx.shape)
            generated_metadata["gp0_shape"] = gp0_qx.shape
            del gp0_qx
            gp1_qx = group2d_tensor(qx.T, gp_sz)  # (num_gp, gp_sz)
            max1 = _max_reduce_except_dim(gp1_qx.abs(), 0)
            generated_metadata["max1"] = max1
            st1 = recon_grouped2d_tensor(max1.expand_as(gp1_qx), qx.T.shape)
            generated_metadata["gp1_shape"] = gp1_qx.shape
            del gp1_qx
            st = torch.min(st0, st1.T)
            del st0, st1
            qx = qx.div(st)
        else:  # rank1
            max_dims = get_sm3_statistics(qx.abs())
            st = _compute_sm3_scale_tensor(max_dims)
            generated_metadata["max_dims"] = max_dims
            generated_metadata["max1"] = None
            qx = qx.div(st)
    elif scale_type == "id":
        generated_metadata["max1"] = None
    else:
        raise NotImplementedError
    generated_metadata["scaled_shape"] = qx.shape
    return qx, generated_metadata


def create_general_qmap(quant_type, bit, signed):
    if bit == 1:
        return torch.Tensor([-1.0, 1.0]) if signed else torch.Tensor([0.0, 1.0])

    if quant_type == "linear":
        return None
    elif quant_type == "nonlinear":
        return create_dynamic_map(signed, bit - 1, bit if signed else bit - 1)
    elif quant_type == "nonlinear-nozero":
        mapping = create_dynamic_map(signed, bit - 1, bit if signed else bit - 1)
        if not signed:
            mapping[0] = mapping[1]
        return mapping
    elif quant_type == "power-1":
        return create_pow_map(bit, signed, 1)
    elif quant_type == "power-2":
        return create_pow_map(bit, signed, 2)
    elif quant_type == "power-3":
        return create_pow_map(bit, signed, 3)
    elif quant_type == "float-point":
        return create_fp8_map(signed, FP_EXPONENT_BIS_MAP[bit], bit)
    else:
        raise ValueError(f"Not support {quant_type} quant type.")


# nonlinear quantization utils
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    if not signed:
        additional_items = 2 * additional_items
    for i in range(max_exponent_bits):
        fraction_items = int(
            (
                2 ** (i + non_sign_bits - max_exponent_bits) + 1
                if signed
                else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
            )
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

        if additional_items > 0:
            boundaries = torch.linspace(0.1, 1, additional_items + 1)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
            if signed:
                data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()
    return torch.Tensor(data)


def create_fp8_map(signed=True, exponent_bits=5, total_bits=8):
    e = exponent_bits
    has_sign = 1 if signed else 0
    precision_bits = total_bits - has_sign - e
    evalues = []
    for i, val in enumerate(range(-((2 ** (exponent_bits - has_sign))), 2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2 ** -(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2**total_bits
    values.sort()
    code = torch.Tensor(values)
    code /= code.max()

    return code


def nonlinear_quant(qx, qmap, b, round_type="sr"):
    def real_nonlinear_quant(qx, qmap, b, stochastic):
        grouped_qx = group_tensor(qx, 2048)
        return ext_quantization.pack_nonlinear(grouped_qx, qmap, b, stochastic)

    qmaplen = len(qmap)
    if round_type == "real-sr":
        idx = real_nonlinear_quant(qx, qmap, b, True)
    elif round_type == "real-nearest":
        idx = real_nonlinear_quant(qx, qmap, b, False)
    elif round_type.startswith("sr"):
        qx.clamp_(qmap[0], qmap[-1])
        floor_idx = ((qx.unsqueeze(-1) >= qmap).sum(dim=-1) - 1).clamp_(0, qmaplen - 1)
        next_idx = (floor_idx + 1).clamp_max_(qmaplen - 1)
        Z = qmap[next_idx] - qmap[floor_idx]
        Z[Z <= 0] = 1.0
        proba = (qx - qmap[floor_idx]) / Z
        proba = torch.bernoulli(proba, generator=lpmm_generator)
        idx = (floor_idx + proba).round_().to(torch.int)
        if round_type == "sr1":
            idx = idx.clamp_min_(1)
        elif round_type == "sr2":
            idx = idx.clamp_min_(2)
    elif round_type == "down":
        idx = ((qx.unsqueeze(-1) >= qmap).sum(dim=-1) - 1).clamp_(0, qmaplen - 1).to(torch.int)
    elif round_type == "up":
        idx = ((qx.unsqueeze(-1) > qmap).sum(dim=-1)).clamp_(0, qmaplen - 1).to(torch.int)
    elif round_type == "nearest":
        diff_tensor = torch.abs(qx.unsqueeze(-1) - qmap)
        idx = torch.argmin(diff_tensor, dim=-1).to(torch.int)
    return idx


def nonlinear_dequant(qx, qmap, b, shape, round_type="sr"):
    if round_type.startswith("real"):
        num_groups = (shape.numel() + 2047) // 2048
        grouped_x = ext_quantization.unpack_nonlinear(qx, qmap, b, num_groups, 2048)
        x = recon_grouped_tensor(grouped_x, shape)
    else:
        x = qmap[qx.to(torch.int64)]
    return x


# group quantization utils
def group_tensor(input: torch.Tensor, gp_sz: int):
    r"""Group tensor into subtensors of size 'gp_sz'"""
    if not gp_sz > 0:
        raise ValueError("group size need to be a positive integer, but found {}".format(gp_sz))

    input_flatten = input.flatten()
    num_features = input_flatten.shape[0]

    # Reshape the tensor into group
    if num_features % gp_sz != 0:
        # Padding
        new_num_features = (num_features // gp_sz + 1) * gp_sz
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten, torch.zeros([delta], dtype=input.dtype, device=input.device)], dim=0)

    input_groups = input_flatten.view(-1, gp_sz)  # num_groups, group_size
    return input_groups


def recon_grouped_tensor(grouped_tensor: torch.Tensor, shape) -> torch.Tensor:
    r"""Reconstruction the tensor to original (or specific) shape"""
    numel = shape.numel()
    recon_flatten = grouped_tensor.flatten()[:numel]
    recon = recon_flatten.view(shape)
    return recon


def group2d_tensor(input: torch.Tensor, gp_sz: int):
    r"""Group tensor into subtensors of size 'gp_sz'"""
    if not gp_sz > 0:
        raise ValueError("group size need to be a positive integer, but found {}".format(gp_sz))
    if input.dim() != 2:
        raise ValueError("")
    C0, C1 = input.shape[0], input.shape[1]
    # Reshape the tensor into group
    if C1 % gp_sz != 0:
        # Padding
        new_num_features = (C1 // gp_sz + 1) * gp_sz
        delta = new_num_features - C1
        input = torch.cat([input, torch.zeros([C0, delta], dtype=input.dtype, device=input.device)], dim=1)
    input_groups = input.reshape(-1, gp_sz)  # num_groups, group_size
    return input_groups


def recon_grouped2d_tensor(grouped_tensor: torch.Tensor, shape) -> torch.Tensor:
    r"""Reconstruction the tensor to original (or specific) shape"""
    return grouped_tensor.reshape(shape[0], -1)[:, : shape[1]]


def _compute_sm3_scale_tensor(max_dims):
    rank = len(max_dims)
    scale_tensor = max_dims[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        scale_tensor = torch.min(scale_tensor, max_dims[i])
    return scale_tensor


def get_sm3_statistics(x, **kwargs):
    qx = x.abs()
    max_dims = []
    for i in range(x.dim()):
        nu_max = _max_reduce_except_dim(qx, i)
        max_dims.append(nu_max)
    return max_dims


def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result


# basic quant utils
def atom_quant(x, scale, maximal, lo, hi, round_type="sr"):
    if scale is None:
        qx = x * maximal
    else:
        qx = x / scale.expand_as(x) * maximal  # scale x to integer unit
    if round_type in ["sr", "real-sr"]:
        eps = torch.rand(qx.size(), generator=lpmm_generator, device=qx.device) - 0.5
        qx = torch.clamp(qx + eps, lo, hi)
        qx = qx.round_().to(torch.int)
    elif round_type == "up":
        qx = torch.clamp(qx, lo, hi)
        qx = qx.ceil_().to(torch.int)
    elif round_type == "down":
        qx = torch.clamp(qx, lo, hi)
        qx = qx.floor_().to(torch.int)
    elif round_type == ["nearest", "real-nearest"]:
        qx = torch.clamp(qx, lo, hi)
        qx = qx.round_().to(torch.int)
    elif round_type == "sr2":
        eps = torch.rand(qx.size(), generator=lpmm_generator, device=qx.device) - 0.5
        qx = torch.clamp(qx + eps, 2, hi)
        qx = qx.round_().to(torch.int)
    elif round_type == "sr1":
        eps = torch.rand(qx.size(), generator=lpmm_generator, device=qx.device) - 0.5
        qx = torch.clamp(qx + eps, 1, hi)
        qx = qx.round_().to(torch.int)
    else:
        raise NotImplementedError
    return qx


def atom_dequant(qx, scale, maximal):
    if scale is None:
        return qx / maximal
    else:
        return qx / maximal * scale.expand_as(qx)


def prepare_quant_boundary(b, signed):
    B = 2 ** (b - 1) - 1
    UB = 2**b - 1
    hi = MRQ = B if signed else UB  # maximal representable quantized integer
    lo = -B if signed else 0
    return MRQ, lo, hi


def create_pow_map(b, signed, p):
    if signed:
        qmap = torch.linspace(-1, 1, (2**b))  # no zero ver.
        # qmap = torch.linspace(-1, 1, (2 ** b) - 1) # less one ver.
        # qmap = torch.linspace(-1, 1, (2 ** b) + 1)[1:] # no minimal ver.
        if p != 1:
            qmap = qmap.sign() * (qmap.abs() ** p)
    else:
        # qmap = torch.linspace(0, 1, 2 ** b) # default ver.
        qmap = torch.linspace(0, 1, (2**b) + 1)[1:]  # no zero ver.
        if p != 1:
            qmap = qmap**p
    return qmap
