import torch

from atorch.kernels import atorch_layer_norm
from atorch.utils.import_util import is_triton_available


class AtorchLayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        if not is_triton_available():
            raise RuntimeError("Triton is not installed. AtorchLayerNorm need it")
        return super().__init__(*args, **kwargs)

    def forward(self, input):
        return atorch_layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
