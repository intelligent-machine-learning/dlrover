import copy
import unittest

import torch

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.module_replace_optimization import ModuleReplaceOptimization, _check_model_params_device
from atorch.modules.transformer.layers import BertAttentionFA, MultiheadAttentionFA
from atorch.normalization import LayerNorm as ApexLayerNorm
from atorch.tests.toy_modules.toy_module import create_model_context

try:
    from transformers.modeling_bert import BertAttention, BertConfig, BertLayer  # 3.5
except (ModuleNotFoundError, ImportError):
    from transformers.models.bert.modeling_bert import BertAttention, BertConfig, BertLayer  # 4.17

from torch.nn import LayerNorm, Module, MultiheadAttention, TransformerEncoderLayer


class TestModule(Module):
    def __init__(self):
        super().__init__()
        bertconf = BertConfig()
        self.layer1 = BertLayer(bertconf)
        self.layer2 = TransformerEncoderLayer(bertconf.hidden_size, bertconf.num_attention_heads)


class ModuleReplaceOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.model_context = create_model_context()
        self.model_context.model = TestModule()
        self.module_replace_config = {
            "HF_BertAttention_FA": {"need_src_module": True},
            "LayerNorm_Apex": {"init_from_attr": True},
        }
        # set fp16 for fa replace not popped
        AutoAccelerateContext.add_ac_attr("half_precision_dtype", {AutoAccelerateContext.counter: torch.float16})

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_module_replace_optimization(self):
        module_replace_optimization = ModuleReplaceOptimization()
        mc = copy.deepcopy(self.model_context)

        self.assertIsInstance(mc.model.layer1.attention, BertAttention)
        self.assertIsInstance(mc.model.layer2.self_attn, MultiheadAttention)
        self.assertIsInstance(mc.model.layer1.output.LayerNorm, LayerNorm)
        self.assertIsInstance(mc.model.layer2.norm1, LayerNorm)

        mc = module_replace_optimization.apply_wrapper(mc, "module_replace", self.module_replace_config)
        self.assertIsInstance(mc.model.layer1.attention, BertAttentionFA)
        self.assertIsInstance(mc.model.layer2.self_attn, MultiheadAttention)
        self.assertIsInstance(mc.model.layer1.output.LayerNorm, ApexLayerNorm)
        self.assertIsInstance(mc.model.layer2.norm1, ApexLayerNorm)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_module_replace_default_optimization(self):
        module_replace_optimization = ModuleReplaceOptimization()
        mc = copy.deepcopy(self.model_context)

        self.assertIsInstance(mc.model.layer1.attention, BertAttention)
        self.assertIsInstance(mc.model.layer2.self_attn, MultiheadAttention)
        self.assertIsInstance(mc.model.layer1.output.LayerNorm, LayerNorm)
        self.assertIsInstance(mc.model.layer2.norm1, LayerNorm)

        # test default replace config
        mc = module_replace_optimization.apply_wrapper(mc, "module_replace", None)
        self.assertIsInstance(mc.model.layer1.attention, BertAttentionFA)
        self.assertIsInstance(mc.model.layer2.self_attn, MultiheadAttentionFA)
        self.assertIsInstance(mc.model.layer1.output.LayerNorm, ApexLayerNorm)
        self.assertIsInstance(mc.model.layer2.norm1, ApexLayerNorm)

    @unittest.skipIf(not torch.cuda.is_available(), "Must have gpu for mutli device test")
    def test_multi_devices_gpu(self):
        # trigger empty replacement
        module_replace_optimization = ModuleReplaceOptimization()
        mc = copy.deepcopy(self.model_context)
        mc.model.to("cuda:0")

        # test default replace config
        # multi-device
        mc.model.layer1.to("cpu")
        mc = module_replace_optimization.apply_wrapper(mc, "module_replace", {})
        post_replacement_devices = _check_model_params_device(mc.model)
        self.assertCountEqual(post_replacement_devices, ["cpu"])
        self.assertTrue(
            list(post_replacement_devices)[0] == "cpu" or list(post_replacement_devices)[0] == torch.device("cpu")
        )


if __name__ == "__main__":
    unittest.main()
