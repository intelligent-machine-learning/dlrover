# coding=utf-8
from __future__ import absolute_import, unicode_literals

import copy
import math
import os
import random
import unittest

import numpy as np
import torch
from pkg_resources import packaging  # type: ignore
from torch.cuda.amp import GradScaler

from atorch.modules.transformer.inject import replace_module
from atorch.modules.transformer.layers import (
    BertAttentionFA,
    CLIPAttentionFA,
    FlashAttnModule,
    LlamaAttentionFA,
    MultiheadAttentionFA,
    _flash_attn_version,
    fa2_with_glm_mask,
    flash_attn_with_mask_bias,
    has_legacy_fa1,
    is_additive_mask_bias_supported_fa1,
    is_pack_glm_mask_supported_fa2,
)

try:
    from transformers.modeling_bert import BertAttention, BertConfig, BertLayer  # 3.5
except (ModuleNotFoundError, ImportError):
    from transformers.models.bert.modeling_bert import BertAttention, BertConfig, BertLayer  # 4.17
    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPTextConfig

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaConfig,
        LlamaDecoderLayer,
        _expand_mask,
        _make_causal_mask,
    )

    _llama_supported_transformers = True
except (ImportError, ModuleNotFoundError):
    _llama_supported_transformers = False

import gc
import time

from torch.nn import MultiheadAttention, TransformerEncoderLayer


def autocast(enabled, dtype=torch.float16):
    # old than torch version 1.9:
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=enabled, dtype=dtype)
    else:  # newer than torch version 1.9:
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} ms".format((end_time - start_time) * 1000))
    print("Max memory used by tensors = {} MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    return end_time - start_time, torch.cuda.max_memory_allocated()


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third", "split"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "split":
        lengths0 = torch.randint(min(128, max_seqlen), max_seqlen + 1, (batch_size // 4 * 3, 1), device=device)
        lengths1 = torch.randint(
            min(max(1, max_seqlen - 20), 128),
            min(max_seqlen, 128) + 1,
            (batch_size - batch_size // 4 * 3, 1),
            device=device,
        )
        lengths = torch.cat([lengths0, lengths1], dim=0)
    padding_mask = torch.arange(max_seqlen, device=device).expand(batch_size, max_seqlen) < lengths
    return padding_mask


def get_additive_glm_mask(glm_mask, q, s_q, b):
    m = q.new_ones((1, s_q, s_q))
    m = torch.tril(m)
    m = m.expand(b, -1, -1)
    ids = torch.arange(s_q, device=glm_mask.device, dtype=glm_mask.dtype).view(1, -1)
    mask = ids < glm_mask.view(-1, 1)
    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    m = m.unsqueeze(1)
    m = (1 - m) * (-60000.0)
    return m


def get_additive_pack_glm_mask(pack_glm_mask, s_q, b):
    additive_mask_lst = []
    for bidx in range(b):
        pack_attention_mask = torch.tril(torch.ones([s_q, s_q]))
        for i in range(pack_glm_mask.shape[-1]):
            startpoint = pack_glm_mask[bidx, 0, i]
            endpoint = pack_glm_mask[bidx, 1, i]
            pack_attention_mask[startpoint:endpoint, startpoint:endpoint] = 1
        additive_mask_lst.append(pack_attention_mask.unsqueeze(0).unsqueeze(0))
    additive_pack_glm_mask = torch.cat(additive_mask_lst, dim=0)
    additive_pack_glm_mask = (1 - additive_pack_glm_mask) * (-60000.0)
    return additive_pack_glm_mask.to(0)


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
class TestFlashAttn(unittest.TestCase):

    seed = 1234

    def setUp(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.set_flags(_benchmark=False, _deterministic=True)

    def assertTensorEqual(self, tensor0, tensor1):
        self.assertListEqual(list(tensor0.shape), list(tensor1.shape))
        self.assertEqual(tensor0.dtype, tensor1.dtype)
        self.assertFalse(bool(torch.any(torch.isnan(tensor0))))
        self.assertFalse(bool(torch.any(torch.isnan(tensor1))))
        allclose = torch.allclose(tensor0, tensor1, atol=1e-2, rtol=1e-2)
        abs_delta = torch.sum(torch.abs(tensor0 - tensor1))
        max_delta = torch.max(torch.abs(tensor0 - tensor1))
        self.assertTrue(
            allclose,
            msg="tensor not allclose abs=%.4f max=%.4f" % (abs_delta, max_delta),
        )

    def assertStateDictEqual(self, sd0, sd1):
        for key, value in sd0.items():
            if isinstance(value, dict):
                self.assertStateDictEqual(value, sd1[key])
            else:
                with self.subTest("StateDict", key=key):
                    self.assertTensorEqual(value, sd1[key])

    def test_flash_attn(self):
        for batch_size in [4, 16, 22]:
            for seq_len in [512, 513, 1024]:
                for dtype in [torch.float16, torch.bfloat16]:
                    self._test_pack_glm_mask(batch_size, seq_len, seq_len, dtype)
        for has_key_padding_mask in [False, True]:
            self._test_famodule_single_q(has_key_padding_mask=has_key_padding_mask)
        for batch_size in [4, 16, 22]:
            for seq_len in [512, 513, 1024]:
                for dtype in [torch.float16, torch.bfloat16]:
                    self._test_fa2_with_glm_mask(batch_size, seq_len, seq_len, dtype)
        for batch_size in [4, 16, 22]:
            for seq_q in [512, 513]:  # test odd seq len
                for seq_k in [512, 513]:
                    self._test_flash_attn_with_mask_bias(batch_size, seq_q, seq_k)
        for batch_size in [1, 4, 16]:
            for seq_len in [32, 128, 512, 1024]:
                self._test_HF_clip_autocast(batch_size, seq_len)
        for batch_size in [1, 4, 16]:
            for seq_len in [32, 128, 512, 1024]:
                self._test_HF_bert_autocast(batch_size, seq_len)
        for batch_size in [1, 4, 16]:
            for seq_len in [32, 128, 512, 1024]:
                self._test_nnMHA_autocast(batch_size, seq_len)
        if _llama_supported_transformers:
            for batch_size in [1, 4, 16]:
                for seq_len in [32, 128, 512, 1024]:
                    self._test_Llama(batch_size, seq_len)

    def _test_pack_glm_mask(self, b, s_q, s_k, dtype):
        print(f"############## test_pack_glm_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}, dtype: {dtype}")
        if not is_pack_glm_mask_supported_fa2():
            print("Pack glm mask supported version FA2 needed.")
            return

        # gen pack_glm_mask
        _max_num_pair = random.randint(1, 10)
        mask_lst = []
        for _ in range(b):
            valid_num = random.randint(0, _max_num_pair)
            if valid_num == 0:
                mask_lst.append(-torch.ones(1, 2, _max_num_pair))
            else:
                idxs = torch.nn.functional.pad(torch.randperm(s_q - 1)[: valid_num * 2 - 1] + 1, (1, 0))
                sorted_idxs = idxs.sort()[0].reshape(valid_num, 2).transpose(0, 1)
                padded_idxs = torch.nn.functional.pad(sorted_idxs, (0, _max_num_pair - valid_num), value=-1)
                mask_lst.append(padded_idxs.reshape(1, 2, _max_num_pair))
        pack_glm_mask = torch.cat(mask_lst, dim=0).to(torch.int32).to(0)
        additive_pack_glm_mask = get_additive_pack_glm_mask(pack_glm_mask, s_q, b)

        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs)).to(0)
        k = torch.randn((b, s_k, nh, hs)).to(0)
        v = torch.randn((b, s_k, nh, hs)).to(0)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)
        fa = FlashAttnModule(causal=True)

        def ref_attn(q, k, v, additive_pack_glm_mask=None):  # ref to glm
            q = q.permute(0, 2, 1, 3)  # [b, nh, s_q, hs]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(hs))
            if additive_pack_glm_mask is not None:
                attention_scores = attention_scores + additive_pack_glm_mask
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            # ignore dropout
            context_layer = torch.matmul(attention_probs, v)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            return context_layer

        ori_out_fp32 = ref_attn(q, k, v, additive_pack_glm_mask)
        (ori_out_fp32 - label).pow(2).mean().backward()
        scaler = GradScaler()
        with autocast(enabled=True, dtype=dtype):
            ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_pack_glm_mask)
            fa_out_autocast = fa(q_fa, k_fa, v_fa, glm_mask=pack_glm_mask)

        scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        q_grad, k_grad, v_grad = [i.grad * scaler._scale for i in [q, k, v]]
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
        assert (q_grad - q_fa.grad).abs().max().item() <= 2 * (
            q_grad - q_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (k_grad - k_fa.grad).abs().max().item() <= 2 * (
            k_grad - k_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (v_grad - v_fa.grad).abs().max().item() <= 2 * (
            v_grad - v_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, additive_pack_glm_mask)
                (ori_out_fp32 - label).pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, additive_pack_glm_mask)
                (ori_out_fp32 - label).pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_pack_glm_mask)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_pack_glm_mask)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    fa_out_autocast = fa(q_fa, k_fa, v_fa, glm_mask=pack_glm_mask)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    fa_out_autocast = fa(q_fa, k_fa, v_fa, glm_mask=pack_glm_mask)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_famodule_single_q(self, has_key_padding_mask=False):
        print(f"############## _test_famodule_single_q, has_key_padding_mask {has_key_padding_mask}")
        b, n, s_kv_cache, h = 4, 32, 512, 2048
        head_dim = h // n

        q = torch.randn(b, n, 1, head_dim, device="cuda")
        k = torch.randn(b, n, s_kv_cache + 1, head_dim, device="cuda")
        v = torch.randn(b, n, s_kv_cache + 1, head_dim, device="cuda")
        if has_key_padding_mask:
            key_padding_mask = torch.randint(0, 2, (b, s_kv_cache + 1), device="cuda")
            key_padding_mask[:, -1] = 1  # single query must be valid
        else:
            key_padding_mask = None
        fa = FlashAttnModule(causal=True)

        def ref_comp(query_states, key_states, value_states, key_padding_mask=None):
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            if key_padding_mask is not None:
                attn_weights += (
                    1 - key_padding_mask.unsqueeze(1).unsqueeze(1).expand(b, n, 1, s_kv_cache + 1)
                ) * -10000.0
            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            return attn_output

        ori_out_fp32 = ref_comp(q, k, v, key_padding_mask)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            ori_out_autocast = ref_comp(q, k, v, key_padding_mask)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            fa_out_autocast = fa(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                key_padding_mask=key_padding_mask,
            )
            fa_out_autocast = fa_out_autocast.transpose(1, 2)

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

    def _test_fa2_with_glm_mask(self, b, s_q, s_k, dtype):
        print(f"############## test_fa2_with_glm_mask, bs: {b}, seq_q: {s_q}, seq_k: {s_k}, dtype: {dtype}")
        _is_flash_attn_2 = _flash_attn_version >= packaging.version.Version("2")
        if not _is_flash_attn_2:
            print("glm mask supported version FA2 needed.")
            return
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs)).to(0)
        k = torch.randn((b, s_k, nh, hs)).to(0)
        v = torch.randn((b, s_k, nh, hs)).to(0)
        glm_mask = torch.randint(s_q // 10, s_q // 10 * 9, (b,), dtype=torch.int32).to(0)
        additive_glm_mask = get_additive_glm_mask(glm_mask, q, s_q, b)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        def ref_attn(q, k, v, additive_glm_mask=None):  # ref to glm
            q = q.permute(0, 2, 1, 3)  # [b, nh, s_q, hs]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(hs))
            if additive_glm_mask is not None:
                attention_scores = attention_scores + additive_glm_mask
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            # ignore dropout
            context_layer = torch.matmul(attention_probs, v)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            return context_layer

        ori_out_fp32 = ref_attn(q, k, v, additive_glm_mask)
        (ori_out_fp32 - label).pow(2).mean().backward()
        scaler = GradScaler()
        with autocast(enabled=True, dtype=dtype):
            ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_glm_mask)
            fa_out_autocast = fa2_with_glm_mask(q_fa, k_fa, v_fa, glm_mask)

        scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        q_grad, k_grad, v_grad = [i.grad * scaler._scale for i in [q, k, v]]
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
        assert (q_grad - q_fa.grad).abs().max().item() <= 2 * (
            q_grad - q_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (k_grad - k_fa.grad).abs().max().item() <= 2 * (
            k_grad - k_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (v_grad - v_fa.grad).abs().max().item() <= 2 * (
            v_grad - v_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, additive_glm_mask)
                (ori_out_fp32 - label).pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, additive_glm_mask)
                (ori_out_fp32 - label).pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_glm_mask)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, additive_glm_mask)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    fa_out_autocast = fa2_with_glm_mask(q_fa, k_fa, v_fa, glm_mask)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True, dtype=dtype):
                    fa_out_autocast = fa2_with_glm_mask(q_fa, k_fa, v_fa, glm_mask)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_flash_attn_with_mask_bias(self, b, s_q, s_k):
        print(f"############## test_flash_attn_with_mask_bias, bs: {b}, seq_q: {s_q}, seq_k: {s_k}")
        nh, hs = 32, 64
        q = torch.randn((b, s_q, nh, hs)).to(0)
        k = torch.randn((b, s_k, nh, hs)).to(0)
        v = torch.randn((b, s_k, nh, hs)).to(0)
        if not is_additive_mask_bias_supported_fa1() and not has_legacy_fa1:
            print("flash_attn_with_mask_bias needs attn mask/bias supported version FlashAttn v1.")
            return
        bool_m = (torch.randn((b, 1, s_q, s_k)) > 0).float().to(0)
        float_m = ((-65504.0) * (1.0 - bool_m)).half()
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        q_copy, k_copy, v_copy = [copy.deepcopy(i) for i in [q, k, v]]
        q_fa, k_fa, v_fa = [copy.deepcopy(i) for i in [q, k, v]]
        label = torch.randn((b, s_q, nh, hs)).to(0)

        def ref_attn(q, k, v, bool_m=None):  # ref to glm
            q = q.permute(0, 2, 1, 3)  # [b, nh, s_q, hs]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(hs))
            if bool_m is not None:
                attention_scores = torch.mul(attention_scores, bool_m)
                attention_scores = attention_scores + (-65504.0) * (1.0 - bool_m)
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            # ignore dropout
            context_layer = torch.matmul(attention_probs, v)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            return context_layer

        ori_out_fp32 = ref_attn(q, k, v, bool_m)
        (ori_out_fp32 - label).pow(2).mean().backward()
        scaler = GradScaler()
        with autocast(enabled=True):
            ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, bool_m)
            fa_out_autocast = flash_attn_with_mask_bias(q_fa, k_fa, v_fa, float_m)

        scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast).abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast).abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast).abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast).abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast).abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        q_grad, k_grad, v_grad = [i.grad * scaler._scale for i in [q, k, v]]
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
        assert (q_grad - q_fa.grad).abs().max().item() <= 2 * (
            q_grad - q_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (k_grad - k_fa.grad).abs().max().item() <= 2 * (
            k_grad - k_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (v_grad - v_fa.grad).abs().max().item() <= 2 * (
            v_grad - v_copy.grad
        ).abs().max().item(), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, bool_m)
                (ori_out_fp32 - label).pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ref_attn(q, k, v, bool_m)
                (ori_out_fp32 - label).pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, bool_m)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ref_attn(q_copy, k_copy, v_copy, bool_m)
                scaler.scale((ori_out_autocast.float() - label).pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = flash_attn_with_mask_bias(q_fa, k_fa, v_fa, float_m)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = flash_attn_with_mask_bias(q_fa, k_fa, v_fa, float_m)
                scaler.scale((fa_out_autocast.float() - label).pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_HF_clip_autocast(self, batch_size, seq_len):
        print(f"############## HF_clip autocast, bs: {batch_size}, seq_len: {seq_len}")
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        dtype = torch.float32
        config = CLIPTextConfig(attention_dropout=0.0)
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = generate_random_padding_mask(seq_len, batch_size, device=device, mode="random")
        _mask = copy.deepcopy(mask)
        mask = (
            torch.zeros_like(mask)
            .to(dtype)
            .masked_fill_(~mask, float("-inf"))
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, 1, mask.shape[-1], 1)
        )
        randn_label = torch.randn_like(hidden_state)
        ori_layer = CLIPEncoderLayer(config)
        fa_layer = copy.deepcopy(ori_layer)
        ori_layer_copy = copy.deepcopy(ori_layer)
        fa_layer = replace_module(fa_layer, CLIPAttention, CLIPAttentionFA, config=config, need_src_module=True)
        ori_layer.to(device)
        fa_layer.to(device)
        ori_layer_copy.to(device)
        self.assertStateDictEqual(ori_layer.state_dict(), CLIPAttentionFA.transform_state_dict(fa_layer.state_dict()))

        ori_out_fp32 = ori_layer(hidden_state, mask, None)[0]
        (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
        with autocast(enabled=True):
            ori_out_autocast = ori_layer_copy(hidden_state, mask, None)[0]
            fa_out_autocast = fa_layer(hidden_state, mask, None)[0]
        scaler = GradScaler()
        scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()

        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # grad comparison on Wqkv
        ori_Wqkv_weight_grad_fp32 = torch.cat(
            [
                ori_layer.self_attn.q_proj.weight.grad,
                ori_layer.self_attn.k_proj.weight.grad,
                ori_layer.self_attn.v_proj.weight.grad,
            ]
        )
        ori_Wqkv_bias_grad_fp32 = torch.cat(
            [
                ori_layer.self_attn.q_proj.bias.grad,
                ori_layer.self_attn.k_proj.bias.grad,
                ori_layer.self_attn.v_proj.bias.grad,
            ]
        )
        ori_Wqkv_weight_grad_autocast = torch.cat(
            [
                ori_layer_copy.self_attn.q_proj.weight.grad,
                ori_layer_copy.self_attn.k_proj.weight.grad,
                ori_layer_copy.self_attn.v_proj.weight.grad,
            ]
        )
        ori_Wqkv_bias_grad_autocast = torch.cat(
            [
                ori_layer_copy.self_attn.q_proj.bias.grad,
                ori_layer_copy.self_attn.k_proj.bias.grad,
                ori_layer_copy.self_attn.v_proj.bias.grad,
            ]
        )
        fa_Wqkv_weight_grad_autocast = fa_layer.self_attn.Wqkv.weight.grad
        fa_Wqkv_bias_grad_autocast = fa_layer.self_attn.Wqkv.bias.grad
        ori_Wqkv_weight_grad_fp32 *= scaler._scale
        ori_Wqkv_bias_grad_fp32 *= scaler._scale
        print(
            f"Original weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        print(
            f"FA weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"FA weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        print(
            f"Original bias grad max diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original bias grad mean diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        print(f"FA bias grad max diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item()}")
        print(
            f"FA bias grad mean diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, mask, None)[0]
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, mask, None)[0]
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, mask, None)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, mask, None)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, mask, None)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, mask, None)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_HF_bert_autocast(self, batch_size, seq_len):
        print(f"############## HF_bert autocast, bs: {batch_size}, seq_len: {seq_len}")
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        dtype = torch.float32
        config = BertConfig(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = generate_random_padding_mask(seq_len, batch_size, device=device, mode="random")
        _mask = copy.deepcopy(mask)
        mask = torch.zeros_like(mask).to(dtype).masked_fill_(~mask, float("-inf")).unsqueeze(1).unsqueeze(1)
        randn_label = torch.randn_like(hidden_state)
        ori_layer = BertLayer(config)
        fa_layer = copy.deepcopy(ori_layer)
        ori_layer_copy = copy.deepcopy(ori_layer)
        fa_layer = replace_module(fa_layer, BertAttention, BertAttentionFA, config=config, need_src_module=True)
        ori_layer.to(device)
        fa_layer.to(device)
        ori_layer_copy.to(device)
        self.assertStateDictEqual(ori_layer.state_dict(), BertAttentionFA.transform_state_dict(fa_layer.state_dict()))

        ori_out_fp32 = ori_layer(hidden_state, mask)[0]
        (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
        with autocast(enabled=True):
            ori_out_autocast = ori_layer_copy(hidden_state, mask)[0]
            fa_out_autocast = fa_layer(hidden_state, mask)[0]
        scaler = GradScaler()
        scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()

        # flash output zero on pad mask, while original bert remain values on mask.
        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # grad comparison on Wqkv
        ori_Wqkv_weight_grad_fp32 = torch.cat(
            [
                ori_layer.attention.self.query.weight.grad,
                ori_layer.attention.self.key.weight.grad,
                ori_layer.attention.self.value.weight.grad,
            ]
        )
        ori_Wqkv_bias_grad_fp32 = torch.cat(
            [
                ori_layer.attention.self.query.bias.grad,
                ori_layer.attention.self.key.bias.grad,
                ori_layer.attention.self.value.bias.grad,
            ]
        )
        ori_Wqkv_weight_grad_autocast = torch.cat(
            [
                ori_layer_copy.attention.self.query.weight.grad,
                ori_layer_copy.attention.self.key.weight.grad,
                ori_layer_copy.attention.self.value.weight.grad,
            ]
        )
        ori_Wqkv_bias_grad_autocast = torch.cat(
            [
                ori_layer_copy.attention.self.query.bias.grad,
                ori_layer_copy.attention.self.key.bias.grad,
                ori_layer_copy.attention.self.value.bias.grad,
            ]
        )
        fa_Wqkv_weight_grad_autocast = fa_layer.attention.Wqkv.weight.grad
        fa_Wqkv_bias_grad_autocast = fa_layer.attention.Wqkv.bias.grad
        ori_Wqkv_weight_grad_fp32 *= scaler._scale
        ori_Wqkv_bias_grad_fp32 *= scaler._scale
        print(
            f"Original weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        print(
            f"FA weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"FA weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        print(
            f"Original bias grad max diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original bias grad mean diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        print(f"FA bias grad max diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item()}")
        print(
            f"FA bias grad mean diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, mask)[0]
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, mask)[0]
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, mask)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, mask)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, mask)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, mask)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_nnMHA_autocast(self, batch_size, seq_len):
        print(f"############## nnMHA autocast, bs: {batch_size}, seq_len: {seq_len}")
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        dtype = torch.float32
        config = BertConfig(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)  # use HFBert default setting
        hidden_state = torch.randn(
            seq_len,  # torch Transformer default batch_first=False
            batch_size,
            config.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = generate_random_padding_mask(seq_len, batch_size, device=device, mode="random")
        _mask = copy.deepcopy(mask).transpose(0, 1)  # [seq_len, bs]
        mask = ~mask  # torch Transformer key_padding_mask use 'True' for padded positions
        randn_label = torch.randn_like(hidden_state)
        ori_layer = TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
        )
        fa_layer = copy.deepcopy(ori_layer)
        ori_layer_copy = copy.deepcopy(ori_layer)
        fa_layer = replace_module(fa_layer, MultiheadAttention, MultiheadAttentionFA, need_src_module=True)
        ori_layer.to(device)
        fa_layer.to(device)
        ori_layer_copy.to(device)
        self.assertStateDictEqual(
            ori_layer.state_dict(), MultiheadAttentionFA.transform_state_dict(fa_layer.state_dict())
        )

        ori_out_fp32 = ori_layer(hidden_state, None, mask)
        (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
        with autocast(enabled=True):
            ori_out_autocast = ori_layer_copy(hidden_state, None, mask)
            fa_out_autocast = fa_layer(hidden_state, None, mask)
        scaler = GradScaler()
        scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()

        # flash output zero on pad mask, while original bert remain values on mask.
        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast)[_mask].abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast)[_mask].abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast)[_mask].abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast)[_mask].abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # grad comparison on Wqkv
        ori_Wqkv_weight_grad_fp32 = ori_layer.self_attn.in_proj_weight.grad
        ori_Wqkv_bias_grad_fp32 = ori_layer.self_attn.in_proj_bias.grad
        ori_Wqkv_weight_grad_autocast = ori_layer_copy.self_attn.in_proj_weight.grad
        ori_Wqkv_bias_grad_autocast = ori_layer_copy.self_attn.in_proj_bias.grad
        fa_Wqkv_weight_grad_autocast = fa_layer.self_attn.Wqkv.weight.grad
        fa_Wqkv_bias_grad_autocast = fa_layer.self_attn.Wqkv.bias.grad
        ori_Wqkv_weight_grad_fp32 *= scaler._scale
        ori_Wqkv_bias_grad_fp32 *= scaler._scale
        print(
            f"Original weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        print(
            f"FA weight grad max diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"FA weight grad mean diff: "
            f"{(ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_weight_grad_fp32 - fa_Wqkv_weight_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_weight_grad_fp32 - ori_Wqkv_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        print(
            f"Original bias grad max diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original bias grad mean diff: "
            f"{(ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        print(f"FA bias grad max diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item()}")
        print(
            f"FA bias grad mean diff: " f"{(ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().mean().item()}"
        )
        assert (ori_Wqkv_bias_grad_fp32 - fa_Wqkv_bias_grad_autocast).abs().max().item() <= 2 * (
            (ori_Wqkv_bias_grad_fp32 - ori_Wqkv_bias_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, None, mask)
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, None, mask)
                (ori_out_fp32 - randn_label)[_mask].pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, None, mask)
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, None, mask)
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, None, mask)
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, None, mask)
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[_mask].pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")

    def _test_Llama(self, batch_size, seq_len):
        print(f"############## Llama autocast, bs: {batch_size}, seq_len: {seq_len}")
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        dtype = torch.float32
        config = LlamaConfig(hidden_size=512, intermediate_size=2048)  # default 4096/11008
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        position_ids = torch.arange(0, seq_len, dtype=torch.long).to(device)[None, :].expand(batch_size, seq_len)
        mask = generate_random_padding_mask(seq_len, batch_size, device=device, mode="random")
        _mask = mask.float()
        _mask = _make_causal_mask((batch_size, seq_len), dtype, device) + _expand_mask(_mask, dtype, seq_len)
        randn_label = torch.randn_like(hidden_state)
        ori_layer = LlamaDecoderLayer(config)
        fa_layer = copy.deepcopy(ori_layer)
        ori_layer_copy = copy.deepcopy(ori_layer)
        fa_layer = replace_module(fa_layer, LlamaAttention, LlamaAttentionFA, need_src_module=True)
        ori_layer.to(device)
        fa_layer.to(device)
        ori_layer_copy.to(device)
        self.assertStateDictEqual(ori_layer.state_dict(), fa_layer.state_dict())

        ori_out_fp32 = ori_layer(hidden_state, _mask, position_ids)[0]
        (ori_out_fp32 - randn_label)[mask].pow(2).mean().backward()
        with autocast(enabled=True):
            ori_out_autocast = ori_layer_copy(hidden_state, _mask, position_ids)[0]
            fa_out_autocast = fa_layer(hidden_state, _mask, position_ids)[0]
        scaler = GradScaler()
        scaler.scale((ori_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()
        scaler.scale((fa_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()

        # flash output zero on pad mask, while original bert remain values on mask.
        print(f"Original output max diff: {(ori_out_fp32 - ori_out_autocast)[mask].abs().max().item()}")
        print(f"Original output mean diff: {(ori_out_fp32 - ori_out_autocast)[mask].abs().mean().item()}")
        print(f"FA output max diff: {(ori_out_fp32 - fa_out_autocast)[mask].abs().max().item()}")
        print(f"FA output mean diff: {(ori_out_fp32 - fa_out_autocast)[mask].abs().mean().item()}")
        assert (ori_out_fp32 - fa_out_autocast)[mask].abs().max().item() <= 2 * (
            (ori_out_fp32 - ori_out_autocast)[mask].abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # grad comparison on Wqkv
        ori_q_weight_grad_fp32 = ori_layer.self_attn.q_proj.weight.grad
        ori_k_weight_grad_fp32 = ori_layer.self_attn.k_proj.weight.grad
        ori_v_weight_grad_fp32 = ori_layer.self_attn.v_proj.weight.grad
        ori_q_weight_grad_autocast = ori_layer_copy.self_attn.q_proj.weight.grad
        ori_k_weight_grad_autocast = ori_layer_copy.self_attn.k_proj.weight.grad
        ori_v_weight_grad_autocast = ori_layer_copy.self_attn.v_proj.weight.grad
        fa_q_weight_grad_fp32 = fa_layer.self_attn.q_proj.weight.grad
        fa_k_weight_grad_fp32 = fa_layer.self_attn.k_proj.weight.grad
        fa_v_weight_grad_fp32 = fa_layer.self_attn.v_proj.weight.grad

        ori_q_weight_grad_fp32 *= scaler._scale
        ori_k_weight_grad_fp32 *= scaler._scale
        ori_v_weight_grad_fp32 *= scaler._scale
        print(
            f"Original q weight grad max diff: "
            f"{(ori_q_weight_grad_fp32 - ori_q_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original k weight grad max diff: "
            f"{(ori_k_weight_grad_fp32 - ori_k_weight_grad_autocast).abs().max().item()}"
        )
        print(
            f"Original v weight grad max diff: "
            f"{(ori_v_weight_grad_fp32 - ori_v_weight_grad_autocast).abs().max().item()}"
        )
        print(f"FA q weight grad max diff: " f"{(ori_q_weight_grad_fp32 - fa_q_weight_grad_fp32).abs().max().item()}")
        print(f"FA k weight grad max diff: " f"{(ori_k_weight_grad_fp32 - fa_k_weight_grad_fp32).abs().max().item()}")
        print(f"FA v weight grad max diff: " f"{(ori_v_weight_grad_fp32 - fa_v_weight_grad_fp32).abs().max().item()}")
        assert (ori_q_weight_grad_fp32 - fa_q_weight_grad_fp32).abs().max().item() <= 2 * (
            (ori_q_weight_grad_fp32 - ori_q_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (ori_k_weight_grad_fp32 - fa_k_weight_grad_fp32).abs().max().item() <= 2 * (
            (ori_k_weight_grad_fp32 - ori_k_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"
        assert (ori_v_weight_grad_fp32 - fa_v_weight_grad_fp32).abs().max().item() <= 2 * (
            (ori_v_weight_grad_fp32 - ori_v_weight_grad_autocast).abs().max().item()
        ), "refer to flash attn, ensure absolute error within two times of original autocast"

        # timer comparison
        if os.environ.get("FA_TIMER_COMPARISON", None) is not None:
            # ori fp32
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, _mask, position_ids)[0]
                (ori_out_fp32 - randn_label)[mask].pow(2).mean().backward()
            start_timer()
            for _ in range(10):
                ori_out_fp32 = ori_layer(hidden_state, _mask, position_ids)[0]
                (ori_out_fp32 - randn_label)[mask].pow(2).mean().backward()
            ori_fp32_time, ori_fp32_mem = end_timer_and_print("ori fp32")
            # ori autocast
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, _mask, position_ids)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    ori_out_autocast = ori_layer_copy(hidden_state, _mask, position_ids)[0]
                scaler.scale((ori_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()
            ori_autocast_time, ori_autocast_mem = end_timer_and_print("ori autocast")
            # fa autocast
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, _mask, position_ids)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()
            start_timer()
            for _ in range(10):
                with autocast(enabled=True):
                    fa_out_autocast = fa_layer(hidden_state, _mask, position_ids)[0]
                scaler.scale((fa_out_autocast.to(dtype) - randn_label)[mask].pow(2).mean()).backward()
            fa_autocast_time, fa_autocast_mem = end_timer_and_print("fa autocast")
            print(
                f"fa autocast time: {fa_autocast_time / ori_fp32_time :.2%} of ori fp32, "
                f"{fa_autocast_time / ori_autocast_time :.2%} of ori autocast."
            )
            print(
                f"fa autocast mem: {fa_autocast_mem / ori_fp32_mem :.2%} of ori fp32, "
                f"{fa_autocast_mem / ori_autocast_mem :.2%} of ori autocast."
            )
        else:
            # still call timer for gc
            start_timer()
            end_timer_and_print("")


if __name__ == "__main__":
    unittest.main()
