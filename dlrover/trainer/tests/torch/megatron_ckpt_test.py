# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import tempfile
import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.elastic_agent.torch.ckpt_saver import CheckpointSaver
from dlrover.trainer.torch.megatron import async_checkpoint
from dlrover.trainer.torch.megatron.async_checkpoint import (
    MegatronCheckpointManager,
    load_latest_checkpoint,
    save_checkpoint_to_memory,
    save_checkpoint_to_storage,
)


def mock_get_tracker_filename(checkpoint_dir):
    return os.path.join(checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MegatrionCheckpointTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()
        mock_func = mock_get_tracker_filename
        async_checkpoint.get_checkpoint_tracker_filename = mock_func

    def tearDown(self) -> None:
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

    def test_ddp_save_load(self):
        os.environ["LOCAL_RANK"] = "0"
        model = SimpleNet()
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.001,
        )
        with tempfile.TemporaryDirectory() as tmpdirname:

            def get_args():
                parser = argparse.ArgumentParser(description="Megatron Test")
                args, _ = parser.parse_known_args()
                args.save = tempfile
                return args

            def save_checkpoint(iteration, model, optimizer, opt_scheduler):
                state_dict = {
                    "iteration": iteration,
                    "model_states": model.state_dict(),
                    "optim_states": optimizer.state_dict(),
                }
                path = os.path.join(
                    tmpdirname, str(iteration), "checkpoint.pt"
                )
                torch.save(state_dict, path)

            def load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                load_arg="load",
                strict=True,
            ):
                path = os.path.join(tmpdirname, "20", "checkpoint.pt")
                state_dict = torch.load(path)
                model.load_state_dict(state_dict["model_states"])
                optimizer.load_state_dict(state_dict["optim_states"])

            async_checkpoint.save_checkpoint = save_checkpoint
            async_checkpoint.load_checkpoint = load_checkpoint
            async_checkpoint.get_args = get_args

            ckpt_manager = MegatronCheckpointManager(tmpdirname)
            save_checkpoint_to_memory(10, model, optimizer, None)
            self.assertFalse(ckpt_manager.engine._shm_handler.empty())
            self.assertEqual(
                ckpt_manager.engine._shm_handler._buffer_size, 9640
            )
            tensor_meta = ckpt_manager.engine._shm_handler._tensor_meta.get()
            self.assertEqual(tensor_meta["iteration"], 10)
            self.assertIsNotNone(tensor_meta["model_states"])
            tracer_file = os.path.join(
                tmpdirname, CheckpointConstant.TRACER_FILE_NAME
            )
            self.assertFalse(os.path.exists(tracer_file))
            ckpt_manager._latest_ckpt_iteration = 10
            ckpt_manager.clear_empty_checkpoint(10)
            self.assertTrue(os.path.exists(tracer_file))
            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
            self.assertEqual(restored_step, 10)

            save_checkpoint_to_storage(20, model, optimizer, None)

            # Wait asynchronously saving.
            while True:
                if "20" in os.listdir(tmpdirname):
                    break
                time.sleep(0.1)

            self.assertTrue(os.path.exists(tracer_file))
            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
            self.assertEqual(restored_step, 20)
            files = os.listdir(tmpdirname + "/20")
            self.assertEqual(files, ["checkpoint.pt"])
            load_latest_checkpoint(model, optimizer, None)
