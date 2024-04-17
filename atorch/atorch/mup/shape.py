# Modifications Copyright 2023 AntGroups, Inc.
# Copyright 2022 Microsoft Corporation.
# MIT license.
# This code is modified from https://github.com/microsoft/mup/blob/main/mup/shape.py.
from copy import deepcopy

import yaml  # type: ignore
from torch import nn
from torch.nn import Linear

from .infshape import InfShape, zip_infshape
from .module import OutputLayer

__BSH_COMMENT__ = """\
# This is a base shape file encoded in yaml
# - a number indicates some dimension of a layer, i.e. some notion of "width"
"""


def get_shapes(model):
    # If you want to implement a custom shapes function, you can use this name
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}


def get_infshapes(model):
    return {name: param.infshape for name, param in model.named_parameters()}


def load_base_shapes(filename):
    """Get a dict of `InfShape` from a filename."""
    with open(filename, "r") as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}


def _extract_shapes(x):
    """
    Input:
        x: can be any of the following:
            - `nn.Module`
            - dict of shapes
            - dict of `InfShape`
            - str of path to a base shapes (.bsh) file
    Output:
        If `x` is dict of `InfShape`, then output itself.
        If `x` is path, then output a dict of `InfShapes` loaded from `x`.
        Else, output the shapes (not `InfShape`) associated to `x`
    """
    if isinstance(x, nn.Module):
        x_shapes = get_shapes(x)
    elif isinstance(x, dict):
        x_shapes = deepcopy(x)
    elif isinstance(x, str):
        # x is file name
        x_shapes = load_base_shapes(x)
    else:
        raise ValueError(f"unhandled x type: {type(x)}")
    return x_shapes


def save_base_shapes(model_or_shapes, file):
    """save base shape object from a model or a dict of shapes."""
    if isinstance(model_or_shapes, nn.Module):
        base_shapes = _extract_shapes(model_or_shapes)
        sh = {k: list(s) for k, s in base_shapes.items()}
    elif isinstance(model_or_shapes, dict):
        shapes = deepcopy(model_or_shapes)
        sh = {k: s.base_shape() for k, s in shapes.items()}
    else:
        raise ValueError()
    s = yaml.dump(sh, None, indent=4)
    s = __BSH_COMMENT__ + s
    if file:
        with open(file, "w") as f:
            f.write(s)


def _dataparallel_hack(base_shapes, shapes):
    """Fix module name discrepancy caused by (Distributed)DataParallel module.

    The parameters of a (Distributed)DataParallel module all have names that
    start with 'module'. This causes a mismatch from non-DataParallel modules.
    This function tries to match `base_shapes` to `shapes`: if the latter starts
    with 'module', then make the former too; likewise if not.
    """
    if all(k.startswith("module.") for k in shapes) and all(not k.startswith("module.") for k in base_shapes):
        return {"module." + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith("module.") for k in shapes) and all(k.startswith("module.") for k in base_shapes):
        return {k.strip("module."): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes


def _zip_infshape_dict(base_shapes, shapes):
    """make a dict of `InfShape` from two dicts of shapes.
    Inputs:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    """
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f"`base_shapes` has extra names {basenames - names}. " f"`shapes` has extra names {names - basenames}."
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes


def zip_infshapes(base, target):
    """make a dict of `InfShape` from models or dicts.
    Inputs:
        base: a base `nn.Module` or a dict of shapes
        target: a target `nn.Module` or a dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    """
    base_shapes = _extract_shapes(base)
    target_shapes = _extract_shapes(target)
    return _zip_infshape_dict(base_shapes, target_shapes)


def clear_dims(infshape_dict):
    """
    Input:
        infshape_dict: dict of `InfShape`
    Output:
        the same dict but where all `InfDim` in all `InfShape`
        have their `dim` attribute set to None
    """
    d = deepcopy(infshape_dict)
    for _, v in d.items():
        for infdim in v:
            infdim.dim = None
    return d


def make_base_shapes(base_shapes, delta_shapes, savefile=None):
    """Make a base shape object from a base model/shapes and a delta model/shapes.

    Inputs:
        base:
            a base `nn.Module` or a dict of shapes
        delta:
            a "delta" model or a dict of shapes, for the sole purpose of
            determining which dimensions are "width" and will be scaled up and
            down in the target model.
        savefile:
            if a string, then the resulting base shape object is serialized to
            this location via yaml encoding.
    Outputs:
        base infshapes
    """
    bsh = clear_dims(zip_infshapes(base_shapes, delta_shapes))
    if savefile is not None:
        save_base_shapes(bsh, savefile)
    return bsh


def apply_infshapes(model, infshapes):
    for name, p in model.named_parameters():
        p.infshape = infshapes[name]


def set_base_shapes(model, base, delta=None, savefile=None, do_assert=True):
    """Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        delta: The delta model used to generate base shapes.
            Can be nn.Module, a dict of shapes, a str, or None.
            We have upgraded this method so that the base shapes can be generated without delta model.
        savefile: a file path to save the base shapes.
        do_assert: decide whether to check the hidden size is infinite.
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    """
    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)
    for name, module in model.named_modules():
        if isinstance(module, OutputLayer):
            module.set_width_mult()
    return model


def assert_hidden_size_inf(model):
    """
    This tests for any `nn.Linear` whose output dimension is finite but input
    dimension is infinite and is not of type `OutputLayer`. Such `nn.Linear`
    modules should not exist in a correctly parametrized models.
    """
    for name, module in model.named_modules():
        if isinstance(module, Linear) and not isinstance(module, OutputLayer):
            if not module.weight.infshape[0].isinf() and module.weight.infshape[1].isinf():
                assert False, (
                    f"{name} has infinite fan-in and finite fan-out dimensions but is not type `OutputLayer`. "
                    "To resolve this, either change the module to `OutputLayer` "
                    "or change the fan-out to an infinite dimension."
                )
