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

import os
import tempfile
import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlrover.python.elastic_agent.torch.ckpt_saver import CheckpointSaver
from dlrover.trainer.torch.deepspeed.async_checkpoint import (
    DeepSpeedCheckpointManger,
)


class MockDeepSpeedEngine(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_engine = None
        self.save_non_zero_checkpoint = True
        self.global_rank = 0

    def zero_optimization(self):
        return False

    def zero_optimization_stage(self):
        return 0

    def save_checkpoint(self, save_dir, tag, client_state, save_latest):
        model_sd = self.model.state_dict()
        model_path = os.path.join(save_dir, tag, "model_states.pt")
        torch.save(model_sd, model_path)
        optimizer_sd = self.optimizer.state_dict()
        optim_path = os.path.join(save_dir, tag, "optim_states.pt")
        torch.save(optimizer_sd, optim_path)

    def load_checkpoint(
        self,
        load_dir,
        tag,
        load_module_strict,
        load_optimizer_states,
        load_lr_scheduler_states,
        load_module_only,
        custom_load_fn,
    ):
        model_path = os.path.join(load_dir, tag, "model_states.pt")
        model_sd = torch.load(model_path)
        self.model.load_state_dict(model_sd)
        optim_path = os.path.join(load_dir, tag, "optim_states.pt")
        optim_sd = torch.load(optim_path)
        self.optimizer.load_state_dict(optim_sd)


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


class DeepSpeedCheckpointTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

    def test_save_load(self):
        os.environ["LOCAL_RANK"] = "0"
        model = SimpleNet()
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.001,
        )
        step = 100
        with tempfile.TemporaryDirectory() as tmpdirname:
            engine = MockDeepSpeedEngine(model, optimizer)
            ckpt_manager = DeepSpeedCheckpointManger(engine, tmpdirname)
            ckpt_manager.save_checkpoint_to_memory(tmpdirname, str(step))
            self.assertFalse(
                ckpt_manager._async_save_engine._shm_handler.empty()
            )
            self.assertEqual(
                ckpt_manager._async_save_engine._shm_handler._buffer_size, 9640
            )
            tensor_meta = (
                ckpt_manager._async_save_engine._shm_handler.metadata.get()
            )
            ds_ckpt_config = tensor_meta["_DLORVER_CKPT_CONFIG"]
            self.assertEqual(ds_ckpt_config.step, str(step))
            model_path = os.path.join(tmpdirname, str(step), "model_states.pt")
            self.assertEqual(ds_ckpt_config.model_path, model_path)
            optim_path = os.path.join(tmpdirname, str(step), "optim_states.pt")
            self.assertEqual(ds_ckpt_config.optimizer_path, optim_path)
            self.assertIsNotNone(tensor_meta["model_states"])
            tracer_file = os.path.join(tmpdirname, "latest")
            self.assertFalse(os.path.exists(tracer_file))

            ckpt_manager.save_checkpoint_to_storage(tmpdirname, str(step))
            # Wait asynchronously saving.
            while True:
                if "100" in os.listdir(tmpdirname):
                    break
                time.sleep(0.1)

            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
            self.assertTrue(os.path.exists(tracer_file))
            self.assertEqual(restored_step, step)

            files = os.listdir(tmpdirname + "/100")
            self.assertEqual(files, ["optim_states.pt", "model_states.pt"])
            ckpt_manager.load_checkpoint(tmpdirname, str(step))
