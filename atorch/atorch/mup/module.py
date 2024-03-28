# Copyright 2023 AntGroups, Inc.
"""
This file defines the `MupModule` and provides some drop-in replacements for `nn.Linear`
to facilitate the implementation of Mup initialization.
"""
from torch import nn

from .init import kaiming_normal_, kaiming_uniform_, normal_, uniform_, xavier_normal_, xavier_uniform_


class MupModule(nn.Module):
    """Drop-in replacement of `nn.Module`.

    Your can easily initialize your model using Maximal Update Parametrization (Mup) via executing `mup_initial` method.

    By setting mode to mup or sp, the model can be initialized with Mup or standard parametrization (SP).
    """

    def mup_initial(self, mode="mup"):
        assert mode in ["mup", "sp"]
        self._mode = mode
        self.apply(self._mup_initial)

    def _mup_initial(self, module):
        if isinstance(module, MupLinear):
            module.mup_initial(self._mode)


class MupLinear(nn.Linear):
    """Drop-in replacement of `nn.Linear` for Mup initialization.

    The initialization take effect only after executing `mup_initial` method.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sampler: sampling method for weight initialization, including uniform, normal, xavier_uniform, xavier_normal,
            kaiming_uniform, and kaiming_normal.
            Default: ``normal``.
        bias_zero_init: If set to ``True``, the bias will be zero initialization.
        init_weight_method: If init_weight_method is not None, the weight will be initialized by init_weight_method.
        init_bias_method: If init_bias_method is not None, the bias will be initialized by init_bias_method.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sampler: str = "normal",
        bias_zero_init: bool = True,
        init_weight_method=None,
        init_bias_method=None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert sampler in ("uniform", "normal", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal")
        self._sampler = sampler
        self._bias_zero_init = bias_zero_init
        self._init_weight_method = init_weight_method
        self._init_bias_method = init_bias_method
        self._kwargs = kwargs

    def mup_initial(self, mode="mup"):
        assert mode in ["mup", "sp"]
        # initialize weight.
        if self._init_weight_method:
            self._init_weight_method(self.weight)
        elif mode == "mup":
            self._mup_init_weight()
        elif mode == "sp":
            self._sp_init_weight()

        # initialize bias. When `init_bias_method` is not given and `bias_zero_init` is False,
        # scale the bias for mup initialization.
        if self.bias is not None:
            if self._init_bias_method:
                self._init_bias_method(self.bias)
            elif self._bias_zero_init:
                self.bias.data.zero_()
            elif mode == "mup":
                self._mup_init_bias()

    def _mup_init_weight(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        if self._sampler == "uniform":
            uniform_(self.weight, a=self._kwargs.get("a", 0.0), b=self._kwargs.get("b", 1.0))
        elif self._sampler == "normal":
            normal_(self.weight, mean=self._kwargs.get("mean", 0.0), std=self._kwargs.get("std", 1.0))
        elif self._sampler == "xavier_uniform":
            xavier_uniform_(self.weight, gain=self._kwargs.get("gain", 1.0))
        elif self._sampler == "xavier_normal":
            xavier_normal_(self.weight, gain=self._kwargs.get("gain", 1.0))
        elif self._sampler == "kaiming_uniform":
            kaiming_uniform_(
                self.weight,
                a=self._kwargs.get("a", 0.0),
                mode=self._kwargs.get("mode", "fan_in"),
                nonlinearity=self._kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif self._sampler == "kaiming_normal":
            kaiming_normal_(
                self.weight,
                a=self._kwargs.get("a", 0.0),
                mode=self._kwargs.get("mode", "fan_in"),
                nonlinearity=self._kwargs.get("nonlinearity", "leaky_relu"),
            )

    def _sp_init_weight(self):
        if self._sampler == "uniform":
            nn.init.uniform_(self.weight, a=self._kwargs.get("a", 0.0), b=self._kwargs.get("b", 1.0))
        elif self._sampler == "normal":
            nn.init.normal_(self.weight, mean=self._kwargs.get("mean", 0.0), std=self._kwargs.get("std", 1.0))
        elif self._sampler == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=self._kwargs.get("gain", 1.0))
        elif self._sampler == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=self._kwargs.get("gain", 1.0))
        elif self._sampler == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                self.weight,
                a=self._kwargs.get("a", 0.0),
                mode=self._kwargs.get("mode", "fan_in"),
                nonlinearity=self._kwargs.get("nonlinearity", "leaky_relu"),
            )
        elif self._sampler == "kaiming_normal":
            nn.init.kaiming_normal_(
                self.weight,
                a=self._kwargs.get("a", 0.0),
                mode=self._kwargs.get("mode", "fan_in"),
                nonlinearity=self._kwargs.get("nonlinearity", "leaky_relu"),
            )

    def _mup_init_bias(self):
        fanin_mult = self.weight.infshape[1].width_mult()
        self.bias.data *= fanin_mult**0.5


class QKVLayer(MupLinear):
    """Drop-in replacement of `nn.Linear` for Mup initialization of QKV matrixes.

    The initialization take effect only after executing `mup_initial` method.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sampler: sampling method for weight initialization, including uniform, normal, xavier_uniform, xavier_normal,
            kaiming_uniform, and kaiming_normal.
            Default: ``normal``.
        q_zero_init: If set to ``True``, the Q matrix will be zero initialization.
        bias_zero_init: If set to ``True``, the bias will be zero initialization.
        init_weight_method: If init_weight_method is not None, the weight will be initialized by init_weight_method.
        init_bias_method: If init_bias_method is not None, the bias will be initialized by init_bias_method.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sampler: str = "normal",
        q_zero_init: bool = True,
        bias_zero_init: bool = True,
        init_weight_method=None,
        init_bias_method=None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            sampler,
            bias_zero_init,
            init_weight_method,
            init_bias_method,
            device,
            dtype,
            **kwargs,
        )
        self._q_zero_init = q_zero_init

    def _mup_init_weight(self):
        super()._mup_init_weight()
        if self._q_zero_init:
            fanout, _ = self.weight.shape
            assert fanout % 3 == 0
            self.weight.data[: fanout // 3, :] = 0.0

    def _sp_init_weight(self):
        super()._sp_init_weight()
        if self._q_zero_init:
            fanout, _ = self.weight.shape
            assert fanout % 3 == 0
            self.weight.data[: fanout // 3, :] = 0.0


class QLayer(QKVLayer):
    """Drop-in replacement of `nn.Linear` for Mup initialization of Q matrix."""

    def _mup_init_weight(self):
        super()._mup_init_weight()
        if self._q_zero_init:
            self.weight.data.zero_()

    def _sp_init_weight(self):
        super()._sp_init_weight()
        if self._q_zero_init:
            self.weight.data.zero_()


class OutputLayer(MupLinear):

    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        output_mult: float = 1.0,
        sampler: str = "normal",
        weight_zero_init: bool = False,
        bias_zero_init: bool = True,
        init_weight_method=None,
        init_bias_method=None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            sampler,
            bias_zero_init,
            init_weight_method,
            init_bias_method,
            device,
            dtype,
            **kwargs,
        )
        self._output_mult = output_mult
        self._weight_zero_init = weight_zero_init
        self._width_mult = 1.0

    def set_width_mult(self):
        """A method to calculate `width_mult` for mup initialization."""
        self._width_mult = self.weight.infshape.width_mult()

    def _mup_init_weight(self):
        """Rescale parameters to convert SP initialization to μP initialization.

        The last layer's initialization is different from the hidden layer.
        The weight is multiplied by a factor `width_mult ** 0.5`.
        """
        super()._sp_init_weight()
        if self._weight_zero_init:
            self.weight.data.zero_()
        else:
            self.weight.data *= self._width_mult**0.5

    def _sp_init_weight(self):
        super()._sp_init_weight()
        if self._weight_zero_init:
            self.weight.data.zero_()

    def _mup_init_bias(self):
        self.bias.data *= self._width_mult**0.5

    def forward(self, x):
        return super().forward(self._output_mult * x / self._width_mult)


class SharedOutputLayer(OutputLayer):
    """`MuReadout` with weights shared with an `nn.Embedding` layer.

    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadout`
    """

    def __init__(self, weight, bias=True, **kwargs):
        super().__init__(*weight.shape, bias=bias, **kwargs)
        self.weight = weight
