import unittest

import pytest

torch = pytest.importorskip("torch", "2.0.0")

from atorch.modules.transformer import losses  # noqa: E402


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
class TestCrossEntropy(unittest.TestCase):
    def test_cross_entropy(self):
        batch = 2
        seq = 1024
        hidden = 32000
        loss_fn_pt = torch.nn.CrossEntropyLoss()
        loss_fn_atorch = losses.CrossEntropyLoss()
        torch.random.manual_seed(0)
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
            with torch.device("cuda"):
                input_gt = torch.randn(batch * seq, hidden).requires_grad_(True)
                input_pt = input_gt.clone().detach().to(dtype).requires_grad_(True)
                input_atorch = input_gt.clone().detach().to(dtype).requires_grad_(True)
                target = torch.empty(batch * seq, dtype=torch.long).random_(hidden)
                loss_pt = loss_fn_pt(input_pt.float(), target)
                loss_atorch = loss_fn_atorch(input_atorch, target)
                loss_pt.backward()
                loss_atorch.backward()
            self.assertTrue(torch.allclose(loss_pt, loss_atorch, rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.allclose(input_pt.grad, input_atorch.grad, rtol=rtol, atol=atol))


if __name__ == "__main__":
    unittest.main()
