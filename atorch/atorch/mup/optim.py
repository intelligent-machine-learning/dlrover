# Modifications Copyright 2023 AntGroups, Inc.
# Copyright 2022 Microsoft Corporation.
# MIT license.
# This code is modified from https://github.com/microsoft/mup/blob/main/mup/optim.py.
"""
Optimizers with μP scaling.

Here we provide 2 ready-to-go optimizers MuAdam and MuSGD.
However, the user can easily convert their own optimizer to a μP
optimizer: if your `optimizer` is "Adam-like", such as RMSProp and Adagrad,
that involves normalizing the gradient entrywise, then the following creates
the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuAdam(params, impl=optimizer, **kwargs)

On the other hand, if your `optimizer` is "SGD-like", such as ASGD, then
the following creates the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuSGD(params, impl=optimizer, **kwargs)

See Appendix B in https://arxiv.org/abs/2203.03466 for discussions of other optimizers.
"""
from collections import defaultdict

from torch.optim import SGD, AdamW


def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)
    return param_groups


def MuAdamParamGroupsAdjust(params, scaled_wd=True, do_assert=True, **kwargs):
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = new_group()
        for p in param_group["params"]:
            if do_assert:
                assert hasattr(p, "infshape"), (
                    f"A parameter with shape {p.shape} does not have `infshape` attribute. "
                    "Did you forget to call `mup.set_base_shapes` on the model?"
                )
            if hasattr(p, "infshape") and p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]["params"].append(p)
            elif hasattr(p, "infshape") and p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 inf dimensions")
            else:
                vector_like_p["params"].append(p)
        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group["lr"] /= width_mult
            if scaled_wd:
                group["weight_decay"] *= width_mult
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    return new_param_groups


def MuAdam(params, impl=AdamW, scaled_wd=True, do_assert=True, **kwargs):
    """Optimizer with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    """
    new_param_groups = MuAdamParamGroupsAdjust(params, scaled_wd=scaled_wd, do_assert=do_assert, **kwargs)
    return impl(new_param_groups, **kwargs)


def MuSGDParamGroupsAdjust(params, scaled_wd=True, do_assert=True, **kwargs):
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        vector_like_p = defaultdict(new_group)  # key is width mult
        matrix_like_p = defaultdict(new_group)  # key is fan_in/out ratio
        fixed_p = new_group()
        for p in param_group["params"]:
            if do_assert:
                assert hasattr(p, "infshape"), (
                    f"A parameter with shape {p.shape} does not have `infshape` attribute. "
                    "Did you forget to call `mup.set_base_shapes` on the model?"
                )
            if hasattr(p, "infshape") and p.infshape.ninf() == 1:
                vector_like_p[p.infshape.width_mult()]["params"].append(p)
            elif hasattr(p, "infshape") and p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]["params"].append(p)
            elif hasattr(p, "infshape") and p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 inf dimensions")
            else:
                fixed_p["params"].append(p)
        for width_mult, group in vector_like_p.items():
            # Scale learning rate and weight decay accordingly
            group["lr"] *= width_mult
            group["weight_decay"] /= width_mult
        for shape_ratio, group in matrix_like_p.items():
            group["lr"] /= shape_ratio
            if scaled_wd:
                group["weight_decay"] *= shape_ratio
        new_param_groups.extend(list(matrix_like_p.values()) + list(vector_like_p.values()) + [fixed_p])
    return new_param_groups


def MuSGD(params, impl=SGD, scaled_wd=True, do_assert=True, **kwargs):
    """SGD with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    """
    new_param_groups = MuSGDParamGroupsAdjust(params, scaled_wd=scaled_wd, do_assert=do_assert, **kwargs)
    return impl(new_param_groups, **kwargs)
