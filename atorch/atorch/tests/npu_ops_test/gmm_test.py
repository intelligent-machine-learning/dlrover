import random
import unittest

import numpy as np
import torch

from atorch.npu.gmm import npu_gmm
from atorch.utils.import_util import is_torch_npu_available


@unittest.skipIf(not is_torch_npu_available(), "NPU is not available")
class TestNPUGroupedGEMM(unittest.TestCase):
    seed = 1234

    def setUp(self):
        from atorch import npu  # noqa

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    @unittest.skipIf(not is_torch_npu_available(), "NPU is not available")
    def test_npu_grouped_gemm(
        self,
    ):
        shapes = [
            (1, 128, 128, 128),
            (8, 128, 128, 128),
            (16, 128, 128, 128),
            (1, 128, 256, 512),
            (8, 128, 256, 512),
            (16, 128, 256, 512),
        ]
        for z, m, k, n in shapes:
            for trans_b in [False, True]:
                for dtype in [torch.float16, torch.bfloat16]:
                    self._test_grouped_gemm_fixed_sizes(z, m, k, n, trans_b, dtype)
                    self._test_grouped_gemm_variable_sizes(z, m, k, n, trans_b, dtype, False)
                    self._test_grouped_gemm_variable_sizes(z, m, k, n, trans_b, dtype, True)

    def _test_grouped_gemm_fixed_sizes(self, z, m, k, n, trans_b, dtype):
        a = torch.randn((z, m, k), dtype=dtype).view(-1, k)
        b = torch.randn((z, n, k), dtype=dtype) if trans_b else torch.randn((z, k, n), dtype=dtype)

        batch_sizes = torch.tensor([m] * z)

        self._test_grouped_gemm(a, b, batch_sizes, trans_b)

    def _test_grouped_gemm_variable_sizes(self, z, m, k, n, trans_b, dtype, batch_sizes_with_zero):
        a = torch.randn((z, m, k), dtype=dtype).view(-1, k)
        b = torch.randn((z, n, k), dtype=dtype) if trans_b else torch.randn((z, k, n), dtype=dtype)

        dist = torch.rand(
            z,
        )
        dist /= dist.sum()
        batch_sizes = (dist * m * z).to(torch.long)
        if batch_sizes_with_zero:
            batch_sizes[random.randint(0, z - 1)] = 0
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        self._test_grouped_gemm(a, b, batch_sizes, trans_b)

    def _test_grouped_gemm(self, a, b, batch_sizes, trans_b):
        a = a.npu(0)
        b = b.npu(0)
        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = npu_gmm(a, b, batch_sizes, trans_b)
        expected_out = self._ref_gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(self._allclose(out, expected_out))

        # Check gradients.
        out.backward(torch.ones(out.shape).npu())
        expected_out.backward(torch.ones(expected_out.shape).npu())
        self.assertTrue(self._allclose(a.grad, a_ref.grad))
        self.assertTrue(self._allclose(b.grad, b_ref.grad))

    def _ref_gmm(self, a, b, batch_sizes, trans_b=False):
        batch_sizes = batch_sizes.numpy()
        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start : start + size, :] @ rhs)
            start += size
        return torch.cat(out)

    def _allclose(self, x, y, pct=2.0):
        mask = torch.isclose(x, y, rtol=1e-5)
        pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
        if pct_diff > pct:
            print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
            print("{:.2f}% of values not close.".format(pct_diff))
            return False
        return True


if __name__ == "__main__":
    unittest.main()
