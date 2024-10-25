# flake8: noqa: E402
import unittest

import pytest

torch = pytest.importorskip("torch", "2.0.0")

from atorch.modules.transformer import rmsnorm
from atorch.utils.import_util import is_triton_available


class LlamaRMSNorm(torch.nn.Module):  # type: ignore[name-defined]
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_triton_available(),
    "cuda or triton is not available",
)
class TestRMSNorm(unittest.TestCase):
    def test_rms_norm(self):
        batch = 2
        seq = 4096
        hidden = 8192
        with torch.device("cuda"):
            rms_llama = LlamaRMSNorm(hidden)
            rms_atorch = rmsnorm.AtorchRmsNorm(hidden)
        torch.random.manual_seed(0)
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        precision = {
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (2e-2, 1e-3),
            torch.float32: (1e-5, 1e-6),
        }
        for dtype in dtypes:
            rtol, atol = precision[dtype]
            with torch.device("cuda"):
                input_gt = torch.randn(batch * seq, hidden).requires_grad_(True)
                input_pt = input_gt.clone().detach().to(dtype).requires_grad_(True)
                input_atorch = input_gt.clone().detach().to(dtype).requires_grad_(True)
                llama_rms_value = rms_llama(input_pt)
                atorch_rms_value = rms_atorch(input_atorch).to(llama_rms_value.dtype)
                atorch_rms_value_same_dtype = atorch_rms_value.to(llama_rms_value.dtype)
                g = torch.randn_like(llama_rms_value)
                llama_grad_dx, llama_grad_dw = torch.autograd.grad(
                    llama_rms_value, (input_pt, rms_llama.weight), grad_outputs=g
                )
                atorch_grad_dx, atorch_grad_dw = torch.autograd.grad(
                    atorch_rms_value, (input_atorch, rms_atorch.weight), grad_outputs=g
                )
                lr = 0.001
                torch.testing.assert_close(llama_rms_value, atorch_rms_value_same_dtype, rtol=rtol, atol=atol)
                torch.testing.assert_close(llama_grad_dx, atorch_grad_dx, rtol=rtol, atol=atol)
                torch.testing.assert_close(llama_grad_dw * lr, atorch_grad_dw * lr, rtol=rtol, atol=atol)
                print(f"{dtype} y max diff: {(llama_rms_value - atorch_rms_value_same_dtype).abs().max().item()}")
                print(f"{dtype} dx max diff: {(llama_grad_dx - atorch_grad_dx).abs().max().item()}")
                print(f"{dtype} dw max diff: {(llama_grad_dw - atorch_grad_dw).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
