# coding=utf-8
from __future__ import absolute_import, unicode_literals

import unittest

import torch
import torch.nn

try:
    from atorch.optimizers.adam_offload import PartitionAdam

    is_apex_available = True
except (ModuleNotFoundError, ImportError):
    is_apex_available = False


@unittest.skipIf(not torch.cuda.is_available(), "Offload optimizer need apex and gpu")
@unittest.skipIf(not is_apex_available, "PartitionAdam import error, need apex and gpu")
class OffloadOptimizerTest(unittest.TestCase):
    def test_optimizer_state_dict(self):
        device = torch.device("cuda")
        model = torch.nn.Linear(10, 20).to(device)
        optimizer = PartitionAdam([{"params": model.parameters()}])

        x = torch.randn(10, 10, device=device)
        y = model(x)
        dy = torch.randn_like(y)
        y.backward(dy)

        optimizer.state_dict()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
