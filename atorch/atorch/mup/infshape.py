# Modifications Copyright 2023 AntGroups, Inc.
# Copyright 2022 Microsoft Corporation.
# MIT license.
# This code is modified from https://github.com/microsoft/mup/blob/main/mup/infshape.py.

from copy import copy


class InfDim:
    """A dimension with a base dimension, used for calculating μP scaling.

    An `InfDim` object is made up of 2 numbers: a dimension and a base
    dimension. If the base dimension is None or equal to the dimension, then
    this object represents a "finite", or "non-width" dimension. Otherwise, it
    represents an "infinite", or "width" dimension.
    """

    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        return self.base_dim is not None and self.base_dim != self.dim

    def width_mult(self):
        """Width multiplier used for calculating μP scaling.

        If finite, return 1.
        If infinite, return dim / base_dim.
        """
        if self.isinf():
            return self.dim / self.base_dim
        return 1

    def __repr__(self):
        return f"InfDim({self.base_dim}, {self.dim})"

    def __str__(self):
        if self.isinf():
            return repr(self)
        return f"FinDim({self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfDim):
            return False
        return self.base_dim == other.base_dim and self.dim == other.dim


class InfShape(tuple):
    """A tuple of `InfDim`s.

    This is intended to be attached to each parameter tensor `p` as `p.infshape`.
    """

    def __init__(self, *args, **kwargs):
        tuple.__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(dim, InfDim):
                raise ValueError("Elements of InfShape needs to be of class InfDim")
        # set main to be the last dimension that is infinite
        # for inf x inf this is fanin
        # for inf x fin or fin x inf it's the unique inf dim
        # user can set this manually if necessary
        self.main_idx = self.main = None
        for i, dim in list(enumerate(self))[::-1]:
            if dim.isinf():
                self.main_idx = i
                self.main = dim
                break

    def fanin_fanout(self):
        assert len(self) >= 2, "fanin, fanout undefined for 1-dimensional weights"
        return self[1], self[0]

    def fanin_fanout_mult_ratio(self):
        fanin, fanout = self.fanin_fanout()
        return fanin.width_mult() / fanout.width_mult()

    def ninf(self):
        return sum(1 for dim in self if dim.isinf())

    def width_mult(self):
        if self.main is not None:
            return self.main.width_mult()
        return 1

    def base_shape(self):
        return [d.base_dim for d in self]

    def shape(self):
        return [d.dim for d in self]

    def __repr__(self):
        r = tuple.__repr__(self)[1:-1]
        return f"InfShape([{r}])"

    def serialize(self):
        d = {"base_shape": [], "shape": []}
        for infdim in self:
            d["shape"].append(infdim.dim)
            d["base_shape"].append(infdim.base_dim)
        return d

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfShape):
            return False
        return all(d == dd for d, dd in zip(self, other))

    @classmethod
    def deserialize(cls, d):
        infshape = []
        for base_dim, dim in zip(d["base_shape"], d["shape"]):
            infshape.append(InfDim(base_dim, dim))
        return InfShape(infshape)

    @classmethod
    def from_base_shape(cls, bsh):
        return InfShape([InfDim(bd, None) for bd in bsh])


def zip_infshape(base_dims, dims, fin_if_same=True):
    infshape = []
    for bd, d in zip(base_dims, dims):
        if isinstance(bd, InfDim):
            # retain bd's base_dim but overwrite dim
            infdim = copy(bd)
            infdim.dim = d
            infshape.append(infdim)
        elif isinstance(bd, int):
            if bd == d and fin_if_same:
                infshape.append(InfDim(None, d))
            else:
                infshape.append(InfDim(bd, d))
        else:
            raise ValueError(f"unhandled base_dim type: {type(bd)}")
    return InfShape(infshape)
