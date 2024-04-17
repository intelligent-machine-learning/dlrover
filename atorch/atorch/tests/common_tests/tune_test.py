import unittest

from atorch.auto.accelerate import model_transform, run_tune_task
from atorch.auto.opt_lib.optimization import Optimization
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.tests.toy_modules.toy_module import create_model_context


class DummyXOptimization(Optimization):
    def __init__(self):
        name = "dummyx"
        group = "dummy"
        is_tunable = True
        super().__init__(name, group, is_tunable)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        best_config = {"batch_size": 10}
        if apply_transform:
            model_context = self.transform(model_context, best_config, apply_wrapper=False)
        return True, best_config, model_context

    def transform(self, model_context, config=None, apply_wrapper=False):
        if model_context.dataloader_args is None:
            model_context.dataloader_args = config
        else:
            model_context.dataloader_args.update(config)
        return model_context


class DummyYOptimization(Optimization):
    def __init__(self):
        name = "dummyy"
        group = "dummy"
        is_tunable = True
        super().__init__(name, group, is_tunable)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        best_config = {"lr": 0.0002}
        if apply_transform:
            model_context = self.transform(model_context, best_config, apply_wrapper=False)
        return True, best_config, model_context

    def transform(self, model_context, config=None, apply_wrapper=False):
        if model_context.optim_args is None:
            model_context.optim_args = config
        else:
            model_context.optim_args.update(config)
        return model_context


class TuneTest(unittest.TestCase):
    def test_tune(self):
        model_context = create_model_context(data_size=8, batch_size=2)
        opt_lib = OptimizationLibrary()
        # add dummy opts
        dummyx = DummyXOptimization()
        dummyy = DummyYOptimization()
        opt_lib.register_opt(dummyx)
        opt_lib.register_opt(dummyy)
        strategy = Strategy()
        strategy.add_opt(("dummyx", None, True))
        strategy.add_opt(("dummyy", None, True))
        strategy.add_opt(("dummyy", {"weight_decay": 0.1}, False))
        status, result = run_tune_task(model_context, strategy, opt_lib)
        self.assertTrue(status)
        # check result
        self.assertTrue(len(result) == 3)
        self.assertTrue(result[0] == ("dummyx", {"batch_size": 10}, False))
        self.assertTrue(result[1] == ("dummyy", {"lr": 0.0002}, False))
        self.assertTrue(result[2] == ("dummyy", {"weight_decay": 0.1}, False))
        mc = model_transform(model_context, result, opt_lib, create_optim=False, create_dataloader=False)
        self.assertTrue(mc.dataloader_args["batch_size"] == 10)
        self.assertTrue(mc.optim_args["lr"] == 0.0002)
        self.assertTrue(mc.optim_args["weight_decay"] == 0.1)
