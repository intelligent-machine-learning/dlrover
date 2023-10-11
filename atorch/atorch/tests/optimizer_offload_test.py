# coding=utf-8
from __future__ import absolute_import, unicode_literals

import unittest

import torch
import torch.nn

try:
    from atorch.optim.adam_offload import PartitionAdam

    is_apex_available = True
except (ModuleNotFoundError, ImportError):
    is_apex_available = False


@unittest.skipIf(not torch.cuda.is_available(), "Offload optimizer need apex and gpu")
@unittest.skipIf(not is_apex_available, "PartitionAdam import error, need apex and gpu")
class OffloadOptimizerTest(unittest.TestCase):
    def test_optimizer_state_dict(self):
        model = torch.nn.Linear(10, 20)

        optimizer = PartitionAdam([{"params": model.parameters()}])
        optimizer.state_dict()


if __name__ == "__main__":
    unittest.main()
