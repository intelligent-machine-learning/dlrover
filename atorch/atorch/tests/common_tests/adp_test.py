# coding=utf-8
from __future__ import absolute_import, unicode_literals

import os
import unittest
from copy import copy

import torch
import torch.nn

from atorch.data_parallel.adp import AllDataParallel


@unittest.skipIf(True, "failed on gpu")  # TODO: fix ut on gpu
# @unittest.skipIf(not torch.cuda.is_available(), "Offload optimizer need apex and gpu")
class OffloadOptimizerTest(unittest.TestCase):
    def wrap_env(self):

        # Shallow copy is enought?
        self.old_env = copy(os.environ)
        new_envs = {
            "MASTER_ADDR": "localhost",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_PORT": "12345",
        }
        os.environ.update(new_envs)

    def unwrap_env(self):
        os.environ = copy(self.old_env)

    def setUp(self) -> None:
        super().setUp()
        self.wrap_env()

    def tearDown(self):
        self.unwrap_env()
        return super().tearDown()

    def test_optimizer_state_dict(self):
        model = torch.nn.ModuleList([torch.nn.Linear(10, 20) for _ in range(5)])
        # CPU backend only
        group = torch.distributed.init_process_group("gloo", world_size=1, rank=0)
        model = AllDataParallel(model, process_group=group)
        # noop, CUDA is required
        # model.state_dict()
        model_str = str(model)
        self.assertIn(
            """ModuleList(
      (0): Linear(in_features=10, out_features=20, bias=True)
      (1): Linear(in_features=10, out_features=20, bias=True)
      (2): Linear(in_features=10, out_features=20, bias=True)
      (3): Linear(in_features=10, out_features=20, bias=True)
      (4): Linear(in_features=10, out_features=20, bias=True)
    )
  )
)""",
            model_str,
        )
        self.assertIn(
            (
                """AllDataParallel(\n  world_size=1, flatten_parameters=True, """
                """mixed_precision=False, reshard_after_forward=True, """
                """data_type=[torch.float32], inject_optimizer=False"""
            ),
            model_str,
        )
        # no... forward need cuda.Stream
        x = torch.Tensor(data=(8, 10, 10))
        model.forward(x)
