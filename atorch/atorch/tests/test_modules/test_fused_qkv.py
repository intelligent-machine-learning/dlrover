import unittest

import torch
from torch.testing import assert_close

from atorch.modules.transformer.linear import AmpFusedDense


class TestQKVLinear(unittest.TestCase):
    device_dtype_combine = [
        ("cpu", torch.float32),
        # ("cpu",torch.float16), # "addmm_impl_cpu_" not implemented for 'Half'
        # ("cpu", torch.bfloat16),  # TODO: install apex from source in aci image
        ("cuda", torch.float32),
        ("cuda", torch.float16),
        # ("cuda", torch.bfloat16),  # TODO: install apex from source in aci image
    ]

    def setUp(self) -> None:
        torch.manual_seed(1241)
        return super().setUp()

    def test_qkv_fused(self):
        for device, dtype in self.device_dtype_combine:

            with self.subTest("test qkv fused", dtypes=dtype, device=device):
                unittest.skipIf(not torch.cuda.is_available() and device == "cuda", "skip cuda test")(
                    self.__test_qkv_fused
                )(dtype, device)

            if device == "cuda":
                with self.subTest("test amp fused", dtypes=dtype, device=device):
                    unittest.skipIf(not torch.cuda.is_available() and device == "cuda", "skip cuda test")(
                        self.__test_amp_fused
                    )(dtype, device=device)
                with self.subTest("test 2d", dtypes=dtype, device=device):
                    unittest.skipIf(not torch.cuda.is_available() and device == "cuda", "skip cuda test")(
                        self.__test_amp_2d
                    )(dtype, device=device)

    def __test_qkv_fused(self, dtype, device):
        bs, seq_length, embed_dim = 10, 500, 128
        x = torch.randn(bs, seq_length, embed_dim, dtype=dtype, device=device)
        wq, wk, wv = (
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        wq.to(device).to(dtype)
        wk.to(device).to(dtype)
        wv.to(device).to(dtype)

        q_origin = wq(x)
        k_origin = wk(x)
        v_origin = wv(x)
        concat_weight = torch.cat([wq.weight.data, wk.weight, wv.weight], dim=0)
        wqkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        wqkv.to(device).to(dtype)
        wqkv.weight.data.copy_(concat_weight)
        concat_bias = torch.cat([wq.bias.data, wk.bias, wv.bias], dim=0)
        wqkv.bias.data.copy_(concat_bias)

        qkv = wqkv(x)

        q, k, v = torch.split(qkv, embed_dim, dim=-1)
        assert_close(q, q_origin)
        assert_close(k, k_origin)
        assert_close(v, v_origin)

    def __test_amp_fused(self, dtype, device):
        bs, seq_length, embed_dim = 10, 500, 128
        x = torch.randn(bs, seq_length, embed_dim, device=device, dtype=dtype)

        wqkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        wqkv.to(device)

        fused_linear = AmpFusedDense(wqkv.in_features, wqkv.out_features, bias=True)
        fused_linear.to(device)
        fused_linear.weight.data.copy_(wqkv.weight)
        fused_linear.bias.data.copy_(wqkv.bias)

        with torch.cuda.amp.autocast(dtype=dtype):
            qkv = wqkv(x)
            out = fused_linear(x)
            assert_close(qkv, out)

    def __test_amp_2d(self, dtype, device):
        bs, seq_length, embed_dim = 10, 500, 128
        x = torch.randn(bs * seq_length, embed_dim, device=device, dtype=dtype)

        wqkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        wqkv.to(device)

        fused_linear = AmpFusedDense(wqkv.in_features, wqkv.out_features, bias=True)
        fused_linear.to(device)
        fused_linear.weight.data.copy_(wqkv.weight)
        fused_linear.bias.data.copy_(wqkv.bias)

        with torch.cuda.amp.autocast(dtype=dtype):
            qkv = wqkv(x)
            out = fused_linear(x)
            assert_close(qkv, out)


if __name__ == "__main__":
    unittest.main()
