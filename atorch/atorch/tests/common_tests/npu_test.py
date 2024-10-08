import copy
import math
import random
import unittest
from itertools import product

import numpy as np
import torch

from atorch.kernels.extensions.npu.grouped_gemm_gmm_npu import npu_gmm
from atorch.npu.layers import AtorchNpuRMSNorm, create_additive_mask_by_breakpoint_mask, npu_fa_with_glm_mask
from atorch.utils.import_util import is_torch_npu_available


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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
    def test_patch_sharded_scaler(self):
        # Disable isort here. Ensure that patching npu ShardedGradScaler before import ShardedGradScaler
        # isort: off
        from atorch import npu  # noqa
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        # isort: on

        self.assertEqual(str(ShardedGradScaler), "<class 'torch_npu.npu.amp.sharded_grad_scaler.ShardedGradScaler'>")


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
                for seq_q, seq_k in [(512, 512), (301, 301)]:  # test odd seq len
                    self._test_npu_fa_with_causal_mask(batch_size, seq_q, seq_k, dtype)
                    for use_additive_mask in (False, True):
                        self._test_npu_fa_with_glm_mask(
                            batch_size, seq_q, seq_k, dtype, use_additive_mask=use_additive_mask
                        )
                        self._test_npu_fa_supported_startpoint_endpoint_mask(
                            batch_size, seq_q, seq_k, dtype, use_additive_mask=use_additive_mask
                        )
                        self._test_npu_fa_gqa_with_glm_mask(
                            batch_size, seq_q, seq_k, dtype, use_additive_mask=use_additive_mask
                        )
                for seq_q, seq_k in [(1, 512), (1, 301)]:
                    self._test_npu_fa_single_query(batch_size, seq_q, seq_k, dtype)

    def _test_npu_fa_supported_startpoint_endpoint_mask(self, b, s_q, s_k, dtype, use_additive_mask=False):
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        glm_mask = self._genenerate_startpoint_endpoint_mask(b, s_q)
        additive_mask = self._create_additive_pack_mask_by_startpoint_endpoint_mask(b, s_q, glm_mask)
        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa1, k_fa1, v_fa1 = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)
        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=additive_mask)
        (ori_out_fp32 - label).pow(2).mean().backward()
        ori_out_lp = self._ref_attn(q_copy.to(dtype), k_copy.to(dtype), v_copy.to(dtype), additive_mask)
        atten_mask = glm_mask if not use_additive_mask else additive_mask
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=atten_mask)
        fa_out1 = npu_fa_with_glm_mask(
            q_fa1.to(dtype),
            k_fa1.to(dtype),
            v_fa1.to(dtype),
            glm_mask=atten_mask,
            breakpoint_additive_mask=additive_mask,
        )
        assert torch.allclose(fa_out, fa_out1), "must be equal"

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

    def _test_npu_fa_with_glm_mask(self, b, s_q, s_k, dtype, use_additive_mask=False):
        print(f"############## test_npu_fa_with_glm_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}")
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        glm_mask = torch.randint(s_q // 8, s_q // 8 * 7, (b,), dtype=torch.int32).to(0)
        additive_mask = create_additive_mask_by_breakpoint_mask(b, s_q, glm_mask)

        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=additive_mask)
        (ori_out_fp32 - label).pow(2).mean().backward()

        ori_out_lp = self._ref_attn(q_copy.to(dtype), k_copy.to(dtype), v_copy.to(dtype), additive_mask)
        atten_mask = glm_mask if not use_additive_mask else additive_mask
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=atten_mask)

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

    def _test_npu_fa_gqa_with_glm_mask(self, b, s_q, s_k, dtype, use_additive_mask=False):
        print(f"############## test_npu_fa_with_glm_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}")
        nh, hs = 32, 64
        nh_kv = 8
        num_key_value_groups = nh // nh_kv
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh_kv, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh_kv, hs), dtype=torch.float32).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        glm_mask = torch.randint(s_q // 8, s_q // 8 * 7, (b,), dtype=torch.int32).to(0)
        additive_mask = create_additive_mask_by_breakpoint_mask(b, s_q, glm_mask)

        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=additive_mask, num_key_value_groups=num_key_value_groups)
        (ori_out_fp32 - label).pow(2).mean().backward()

        ori_out_lp = self._ref_attn(
            q_copy.to(dtype),
            k_copy.to(dtype),
            v_copy.to(dtype),
            additive_mask,
            num_key_value_groups=num_key_value_groups,
        )
        atten_mask = glm_mask if not use_additive_mask else additive_mask
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=atten_mask)

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

    def _test_npu_fa_single_query(self, b, s_q, s_k, dtype):
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs), dtype=torch.float32).to(0)
        k = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        v = torch.randn((b, s_k, nh, hs), dtype=torch.float32).to(0)
        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        ori_out_fp32 = self._ref_attn(q, k, v, bool_m=None)
        ori_out_lp = self._ref_attn(q_copy.to(dtype), k_copy.to(dtype), v_copy.to(dtype), bool_m=None)
        fa_out = npu_fa_with_glm_mask(q_fa.to(dtype), k_fa.to(dtype), v_fa.to(dtype), glm_mask=None)
        assert (ori_out_fp32 - fa_out).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_lp).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

    def _ref_attn(self, q, k, v, bool_m=None, num_key_value_groups=None):  # ref to glm
        q = q.permute(0, 2, 1, 3)  # [b, nh, s_q, hs]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        hs = q.shape[3]
        if num_key_value_groups is not None:
            k = self.repeat_kv(k, num_key_value_groups)
            v = self.repeat_kv(v, num_key_value_groups)
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

    def _create_additive_pack_mask_by_startpoint_endpoint_mask(self, batch_size, seq_length, pack_glm_mask):
        assert pack_glm_mask.dim() == 3
        additive_mask_lst = []
        for bidx in range(batch_size):
            # start_ends = np.array(pack_glm_mask.cpu())
            start_ends = pack_glm_mask
            starts = start_ends[bidx, 0, :]
            ends = start_ends[bidx, 1, :]

            # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
            m = torch.ones((seq_length, seq_length), device=pack_glm_mask.device).tril()
            for i in range(len(starts)):
                m[starts[i] :, : ends[i]] = 1

            additive_mask_lst.append(m.unsqueeze(0).unsqueeze(0))
        additive_mask_lst = torch.cat(additive_mask_lst, dim=0)

        return additive_mask_lst

    def _genenerate_startpoint_endpoint_mask(self, b, s_q):
        _max_num_pair = random.randint(2, 10)
        mask_lst = []
        for _ in range(b):
            valid_num = torch.randint(1, _max_num_pair, (), device=0)
            if valid_num == 0:
                mask_lst.append(-torch.ones(1, 2, _max_num_pair, device=0))
            else:
                idxs = torch.nn.functional.pad(torch.randperm(s_q - 1, device=0)[: valid_num * 2 - 1] + 1, (1, 0))
                sorted_idxs = idxs.sort()[0].reshape(valid_num, 2).transpose(0, 1)
                padded_idxs = torch.nn.functional.pad(sorted_idxs, (0, _max_num_pair - valid_num), value=-1)
                mask_lst.append(padded_idxs.reshape(1, 2, _max_num_pair))
        pack_glm_mask = torch.cat(mask_lst, dim=0).to(torch.int32)
        return pack_glm_mask

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@unittest.skipIf(not is_torch_npu_available(), "NPU is not available")
class TestNPUNorm(unittest.TestCase):
    seed = 1234

    def setUp(self):
        from atorch import npu  # noqa

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    def test_npu_fused_rms_norm(self):
        b, s = 3, 512
        device = "npu:0"
        for h in (1024, 2048, 4096, 5120, 6144, 12288):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                npu_fused_rms_norm = AtorchNpuRMSNorm(h).to(dtype).to(device)
                rms_norm = RMSNorm(h).to(dtype).to(device)
                hidden_states = torch.randn((b, s, h), dtype=dtype, device=device)
                fused_output = npu_fused_rms_norm(hidden_states)
                normal_output = rms_norm(hidden_states)
                self.assertTrue(torch.allclose(fused_output, normal_output, rtol=5e-3))


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
                    self._test_grpuped_gemm_fixed_sizes(z, m, k, n, trans_b, dtype)
                    self._test_grpuped_gemm_variable_sizes(z, m, k, n, trans_b, dtype)

    def _test_grpuped_gemm_fixed_sizes(self, z, m, k, n, trans_b, dtype):
        a = torch.randn((z, m, k), dtype=dtype).view(-1, k)
        b = torch.randn((z, n, k), dtype=dtype) if trans_b else torch.randn((z, k, n), dtype=dtype)

        batch_sizes = torch.tensor([m] * z)

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

    def _test_grpuped_gemm_variable_sizes(self, z, m, k, n, trans_b, dtype):
        a = torch.randn((z, m, k), dtype=dtype).view(-1, k)
        b = torch.randn((z, n, k), dtype=dtype) if trans_b else torch.randn((z, k, n), dtype=dtype)

        dist = torch.rand(
            z,
        )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

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
