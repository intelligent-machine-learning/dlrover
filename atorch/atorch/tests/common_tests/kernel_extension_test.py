import unittest

import torch

from atorch.kernels.extensions.flash_attention.dropout_add_layer_norm import DropoutAddLayernormExtension
from atorch.kernels.extensions.flash_attention.flash_attn_cross_entropy import FlashAttnCrossEntropyExtension
from atorch.kernels.extensions.flash_attention.flash_attn_func_ext import FlashAttnFuncExtension
from atorch.kernels.extensions.flash_attention_1.dropout_add_layer_norm_1 import DropoutAddLayernorm1Extension
from atorch.kernels.extensions.flash_attention_1.flash_attn_func_ext_1 import FlashAttnFunc1Extension
from atorch.kernels.extensions.grouped_gemm_exts.grouped_gemm_gmm import GroupedGEMMExtension
from atorch.kernels.extensions.npu.adamw_npu import FusedAdamwNpuExtension
from atorch.kernels.extensions.npu.flash_attention_npu import FlashAttentionNpuExtension
from atorch.utils.fa_util import patch_fa_interface_to_autocast


@unittest.skipIf(not torch.cuda.is_available(), "test kernel extensions on gpu environment")
class KernelExtensionTest(unittest.TestCase):
    def test_flash_attn_dropout_add_layernorm_extension(self):
        has_ext = False
        try:
            import flash_attn  # noqa F401
            from flash_attn.ops.layer_norm import dropout_add_layer_norm  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = DropoutAddLayernormExtension()
        if has_ext:
            from flash_attn.ops.layer_norm import dropout_add_layer_norm  # noqa

            assert ext.is_available() and ext.load() == dropout_add_layer_norm
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_flash_attn_cross_entropy_extension(self):
        has_ext = False
        try:
            import flash_attn  # noqa F401
            from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = FlashAttnCrossEntropyExtension()
        if has_ext:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss  # noqa

            assert ext.is_available() and ext.load() == CrossEntropyLoss
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_flash_attn_func_extension(self):
        has_ext = False
        try:
            import flash_attn  # noqa F401
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = FlashAttnFuncExtension()
        if has_ext:
            import flash_attn.flash_attn_interface  # noqa

            patch_fa_interface_to_autocast(flash_attn.flash_attn_interface)

            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa

            assert ext.is_available() and ext.load() == (flash_attn_func, flash_attn_varlen_func)
        else:
            a, b = ext.load()
            assert (not ext.is_available()) and a is None and b is None

    def test_flash_attn_1_dropout_add_layernorm_extension(self):
        has_ext = False
        try:
            import flash_attn_1  # noqa F401
            from flash_attn_1.ops.layer_norm import dropout_add_layer_norm  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = DropoutAddLayernorm1Extension()
        if has_ext:
            from flash_attn_1.ops.layer_norm import dropout_add_layer_norm  # noqa

            assert ext.is_available() and ext.load() == dropout_add_layer_norm
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_flash_attn_1_func_extension(self):
        has_ext = False
        try:
            import flash_attn_1  # noqa F401
            from flash_attn_1.flash_attn_interface import flash_attn_unpadded_func  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = FlashAttnFunc1Extension()
        if has_ext:
            import flash_attn_1.flash_attn_interface

            patch_fa_interface_to_autocast(flash_attn_1.flash_attn_interface)

            from flash_attn_1.flash_attn_interface import flash_attn_unpadded_func  # noqa

            assert ext.is_available() and ext.load() == flash_attn_unpadded_func
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_grouped_gemm_gmm_extension(self):
        has_ext = False
        try:
            import grouped_gemm  # noqa

            has_ext = True
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = GroupedGEMMExtension()
        if has_ext:
            from atorch.kernels.extensions.grouped_gemm_exts.grouped_gemm_gmm import _cast_fn

            gmm, gg = ext._load_with_ext_package()
            assert ext.is_available() and gmm == _cast_fn(gg.ops.gmm)
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_adamw_npu_extension(self):
        has_ext = False
        try:
            import torch_npu  # noqa

            has_ext = hasattr(torch_npu, "npu_apply_adam_w")
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = FusedAdamwNpuExtension()
        if has_ext:
            from torch_npu import npu_apply_adam_w

            assert ext.is_available() and ext.load() == npu_apply_adam_w
        else:
            assert (not ext.is_available()) and ext.load() is None

    def test_flash_attention_npu_extension(self):
        has_ext = False
        try:
            import torch_npu  # noqa

            has_ext = hasattr(torch_npu, "npu_fusion_attention")
        except (ImportError, ModuleNotFoundError):
            has_ext = False

        ext = FlashAttentionNpuExtension()
        if has_ext:
            from torch_npu import npu_fusion_attention

            assert ext.is_available() and ext.load() == npu_fusion_attention
        else:
            assert (not ext.is_available()) and ext.load() is None
