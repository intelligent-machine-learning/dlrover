import copy
import math
import random
import unittest
from itertools import product

import numpy as np
import torch

from atorch.npu.layers import create_additive_mask_by_glm_mask, npu_fa_with_glm_mask
from atorch.utils.import_util import is_torch_npu_available


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

    @unittest.skipIf(not torch.cuda.is_available(), "we test npu patch on gpu/npu environment")
    def test_bf16_checking(self):
        # can execute on gpu environment
        from transformers.utils import is_torch_bf16_gpu_available

        from atorch import npu  # noqa

        device_support_bf16 = is_torch_bf16_gpu_available()
        device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
        if not is_torch_npu_available():
            if device_property.major < 8:
                self.assertFalse(device_support_bf16)
            if int(torch.version.cuda.split(".")[0]) < 11:
                self.assertFalse(device_support_bf16)
        else:
            if device_property.name == "Ascend910B2":
                self.assertTrue(device_support_bf16)


@unittest.skipIf(not is_torch_npu_available(), "NPU is not available")
class TestNPUFlashAttn(unittest.TestCase):
    seed = 1234

    def setUp(self):
        from atorch import npu  # noqa

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    def test_npu_flash_attn(self):
        for dtype in [torch.float16, torch.bfloat16]:
            for batch_size in [5, 8, 17]:
                for (seq_q, seq_k) in [(512, 512), (301, 301)]:  # test odd seq len
                    self._test_npu_fa_with_glm_mask(batch_size, seq_q, seq_k, dtype)
                    self._test_npu_fa_with_causal_mask(batch_size, seq_q, seq_k, dtype)

    def _test_npu_fa_with_glm_mask(self, b, s_q, s_k, dtype):
        print(f"############## test_npu_fa_with_glm_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}")
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        glm_mask = torch.randint(s_q // 8, s_q // 8 * 7, (b,), dtype=torch.int32).to(0)
        additive_mask = create_additive_mask_by_glm_mask(b, s_q, q, glm_mask)

        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=additive_mask)
        (ori_out_fp32 - label).pow(2).mean().backward()

        ori_out_lp = self._ref_attn(q_copy.to(dtype), k_copy.to(dtype), v_copy.to(dtype), additive_mask)
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=glm_mask)

        (ori_out_lp.float() - label).pow(2).mean().backward()
        (fa_out.float() - label).pow(2).mean().backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_lp).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_lp).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_lp).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        q_grad, k_grad, v_grad = [i.grad for i in [q, k, v]]
        print(f"Original q grad max diff: {(q_grad - q_copy.grad).abs().max().item()}")
        print(f"Original k grad max diff: {(k_grad - k_copy.grad).abs().max().item()}")
        print(f"Original v grad max diff: {(v_grad - v_copy.grad).abs().max().item()}")
        print(f"Original q grad mean diff: {(q_grad - q_copy.grad).abs().mean().item()}")
        print(f"Original k grad mean diff: {(k_grad - k_copy.grad).abs().mean().item()}")
        print(f"Original v grad mean diff: {(v_grad - v_copy.grad).abs().mean().item()}")
        print(f"FA q grad max diff: {(q_grad - q_fa.grad).abs().max().item()}")
        print(f"FA k grad max diff: {(k_grad - k_fa.grad).abs().max().item()}")
        print(f"FA v grad max diff: {(v_grad - v_fa.grad).abs().max().item()}")
        print(f"FA q grad mean diff: {(q_grad - q_fa.grad).abs().mean().item()}")
        print(f"FA k grad mean diff: {(k_grad - k_fa.grad).abs().mean().item()}")
        print(f"FA v grad mean diff: {(v_grad - v_fa.grad).abs().mean().item()}")
        assert (q_grad - q_fa.grad).abs().max().item() <= 2 * (q_grad - q_copy.grad).abs().max().item() or (
            q_grad - q_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (k_grad - k_fa.grad).abs().max().item() <= 2 * (k_grad - k_copy.grad).abs().max().item() or (
            k_grad - k_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (v_grad - v_fa.grad).abs().max().item() <= 2 * (v_grad - v_copy.grad).abs().max().item() or (
            v_grad - v_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"

    def _test_npu_fa_with_causal_mask(self, b, s_q, s_k, dtype):
        print(f"############## _test_npu_fa_with_causal_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}")
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        causal_mask = self._make_causal_mask(s_q)

        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=causal_mask.to(0))
        (ori_out_fp32 - label).pow(2).mean().backward()

        ori_out_lp = self._ref_attn(q_copy.to(dtype), k_copy.to(dtype), v_copy.to(dtype), bool_m=causal_mask.to(0))
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=None, causal=True)

        (ori_out_lp.float() - label).pow(2).mean().backward()
        (fa_out.float() - label).pow(2).mean().backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_lp).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_lp).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_lp).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        q_grad, k_grad, v_grad = [i.grad for i in [q, k, v]]
        print(f"Original q grad max diff: {(q_grad - q_copy.grad).abs().max().item()}")
        print(f"Original k grad max diff: {(k_grad - k_copy.grad).abs().max().item()}")
        print(f"Original v grad max diff: {(v_grad - v_copy.grad).abs().max().item()}")
        print(f"Original q grad mean diff: {(q_grad - q_copy.grad).abs().mean().item()}")
        print(f"Original k grad mean diff: {(k_grad - k_copy.grad).abs().mean().item()}")
        print(f"Original v grad mean diff: {(v_grad - v_copy.grad).abs().mean().item()}")
        print(f"FA q grad max diff: {(q_grad - q_fa.grad).abs().max().item()}")
        print(f"FA k grad max diff: {(k_grad - k_fa.grad).abs().max().item()}")
        print(f"FA v grad max diff: {(v_grad - v_fa.grad).abs().max().item()}")
        print(f"FA q grad mean diff: {(q_grad - q_fa.grad).abs().mean().item()}")
        print(f"FA k grad mean diff: {(k_grad - k_fa.grad).abs().mean().item()}")
        print(f"FA v grad mean diff: {(v_grad - v_fa.grad).abs().mean().item()}")
        assert (q_grad - q_fa.grad).abs().max().item() <= 2 * (q_grad - q_copy.grad).abs().max().item() or (
            q_grad - q_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (k_grad - k_fa.grad).abs().max().item() <= 2 * (k_grad - k_copy.grad).abs().max().item() or (
            k_grad - k_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (v_grad - v_fa.grad).abs().max().item() <= 2 * (v_grad - v_copy.grad).abs().max().item() or (
            v_grad - v_fa.grad
        ).abs().max().item() < 1e-5, "refer to flash attn, ensure absolute error within two times of original autocast"

    def _ref_attn(self, q, k, v, bool_m=None):  # ref to glm
        q = q.permute(0, 2, 1, 3)  # [b, nh, s_q, hs]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        hs = q.shape[3]
        attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(hs))
        if bool_m is not None:
            attention_scores = torch.mul(attention_scores, bool_m)
            attention_scores = attention_scores + (-65504.0) * (1.0 - bool_m)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # ignore dropout
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer

    def _make_causal_mask(self, seqlen):
        q_indices = torch.arange(seqlen) - seqlen
        kv_indices = torch.arange(seqlen) - seqlen
        causal_mask_bool = q_indices.view(-1, 1) >= kv_indices.view(1, -1)
        return causal_mask_bool.to(torch.int32)


if __name__ == "__main__":
    unittest.main()
