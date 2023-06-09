import unittest

import torch

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.model_context import ModelContext
from atorch.tests.toy_module import ToyModel


class AutoAccContextTest(unittest.TestCase):
    def test_context(self):
        model = ToyModel()
        mc = ModelContext(
            model=model,
            optim_func=torch.optim.Adam,
            optim_args={"lr": 1e-3},
        )
        mc.create_optim()
        loss = torch.tensor(1, dtype=torch.int32)

        # Add some attributes
        AutoAccelerateContext.add_ac_attr("a", 1)
        AutoAccelerateContext.add_ac_attr("b", [1, 2])
        AutoAccelerateContext.add_ac_attr("c", {3: 4})
        AutoAccelerateContext.add_ac_attr("fn", lambda x: x + 1)
        AutoAccelerateContext.add_ac_attr("model", mc.model)
        AutoAccelerateContext.add_ac_attr("optim", mc.optim)
        AutoAccelerateContext.add_ac_attr("loss", loss)

        # check attributes
        self.assertEqual(AutoAccelerateContext.a, 1)
        self.assertEqual(AutoAccelerateContext.b, [1, 2])
        self.assertEqual(AutoAccelerateContext.c, {3: 4})
        self.assertEqual(AutoAccelerateContext.fn(0), 1)
        self.assertEqual(AutoAccelerateContext.model, mc.model)
        self.assertEqual(AutoAccelerateContext.optim, mc.optim)
        self.assertTrue(torch.equal(AutoAccelerateContext.loss, loss))

        # reset context to delete all attributes add by `add_ac_attr`
        AutoAccelerateContext.reset()
        self.assertFalse(hasattr(AutoAccelerateContext, "a"))
        self.assertFalse(hasattr(AutoAccelerateContext, "b"))
        self.assertFalse(hasattr(AutoAccelerateContext, "c"))
        self.assertFalse(hasattr(AutoAccelerateContext, "fn"))
        self.assertFalse(hasattr(AutoAccelerateContext, "model"))
        self.assertFalse(hasattr(AutoAccelerateContext, "optim"))
        self.assertFalse(hasattr(AutoAccelerateContext, "loss"))

        # class method still exists after calling `reset`
        self.assertTrue(hasattr(AutoAccelerateContext, "add_ac_attr"))
        self.assertTrue(hasattr(AutoAccelerateContext, "reset"))
