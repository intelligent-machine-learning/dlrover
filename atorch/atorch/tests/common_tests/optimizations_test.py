import copy
import unittest

import torch

from atorch.auto.accelerate import model_transform
from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.tests.toy_modules.toy_module import create_model_context, run_train


class OptimizationsTest(unittest.TestCase):
    def setUp(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        data_size = 8
        batch_size = 2
        model_context = create_model_context(data_size=data_size, batch_size=batch_size, use_optim_param_func=True)
        model_context.model.to(device)
        self.mc = model_context
        self.data_size = data_size
        self.batch_size = batch_size
        self.opt_lib = OptimizationLibrary()
        self.device = device

    def run_test(self, mc, strategy, input_dtype=torch.float32):
        model_context = model_transform(mc, strategy, self.opt_lib)
        num = run_train(
            model_context.model,
            model_context.dataloader,
            model_context.optim,
            model_context.prepare_input,
            model_context.loss_func,
            self.device,
            input_dtype=input_dtype,
        )
        self.assertEqual(num, self.data_size // self.batch_size)

    def test_checkpoint(self):
        module_type = "ToyCustomModule"
        mc = copy.deepcopy(self.mc)
        strategy = Strategy([("checkpoint", module_type, False)])
        self.run_test(mc, strategy)

    @unittest.skipIf(not torch.cuda.is_available(), "half is only supported by gpu")
    def test_half(self):
        mc = copy.deepcopy(self.mc)
        config = "fp16"
        strategy = Strategy([("half", config, False)])
        self.run_test(mc, strategy, torch.float16)

        AutoAccelerateContext.counter += 1
        mc = copy.deepcopy(self.mc)
        config = "bf16"
        strategy = Strategy([("half", config, False)])
        self.run_test(mc, strategy, torch.bfloat16)
