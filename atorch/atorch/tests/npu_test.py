import unittest
from itertools import product

import torch


class NPUPatchTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "we test npu patch on gpu/npu environment")
    def test_npu_patch(self):
        # can execute on gpu environment
        from atorch import npu  # noqa

        device = torch.device("cuda")
        self.assertIsNotNone(torch.cuda.get_device_capability(device))
        devices = [0, "cuda", None, device]
        for device in devices:
            self.assertIsNotNone(npu.new_device_capability(device))

    @unittest.skipIf(not torch.cuda.is_available(), "we test npu patch on gpu/npu environment")
    def test_empty(self):
        # can execute on gpu environment
        from atorch import npu  # noqa

        tensor = torch.randn(10, 10, device="cuda", dtype=torch.float32)
        tensor_16 = torch.empty_like(tensor, dtype=torch.float16)
        self.assertEqual(tensor_16.dtype, torch.float16)

    @unittest.skipIf(not torch.cuda.is_available(), "we test npu patch on gpu/npu environment")
    def test_vertor_norm(self):
        # can execute on gpu environment
        from atorch import npu  # noqa

        tensor = torch.empty((10, 10), dtype=torch.float16, device="cuda")
        torch.linalg.vector_norm(tensor, 2, dtype=torch.float32)

    def test_dtype_diff(self):
        # scan torch function, find is there have dtype args
        dtypes = [torch.float16, torch.float32, torch.float64]
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for from_dtype, to_dtype in product(dtypes, dtypes):
            with self.subTest("dtype diff", from_dtype=from_dtype, to_dtype=to_dtype):
                tensor = torch.randn(10, 10, dtype=from_dtype)
                tensor_t = torch.empty_like(tensor, dtype=to_dtype)
                self.assertEqual(tensor_t.dtype, to_dtype)

                if torch.cuda.is_available():
                    from atorch import npu  # noqa

                    tensor = torch.randn(
                        10,
                        10,
                        dtype=from_dtype,
                        device="cuda",
                    )
                    tensor_t = torch.empty_like(tensor, dtype=to_dtype)
                    self.assertEqual(tensor_t.dtype, to_dtype)
