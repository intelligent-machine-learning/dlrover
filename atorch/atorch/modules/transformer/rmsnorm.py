import torch

from atorch.utils.import_util import is_torch_npu_available, is_triton_available

if is_triton_available():
    from atorch.kernels.triton_jit.rmsnorm_kernel import AtorchRmsNormFunc

if is_torch_npu_available():
    from atorch.kernels import npu_rms_norm


class AtorchRmsNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-06, dtype=torch.float32, reset_fn=None):
        if not is_triton_available() and not is_torch_npu_available():
            raise NotImplementedError("No backend found, AtorchRmsNorm is not available")
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.dtype = dtype
        self.reset_fn = reset_fn

    def forward(self, x):
        if is_torch_npu_available():
            return npu_rms_norm(x, self.weight, self.variance_epsilon)[0]
        return AtorchRmsNormFunc.apply(x, self.weight, self.variance_epsilon)

    def reset_parameters(self):
        if self.reset_fn is not None:
            self.reset_fn(self.weight)
            return
        torch.nn.init.ones_(self.weight)
