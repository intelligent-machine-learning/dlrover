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

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.multi_process import clear_sock_dir
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DeepSpeedCheckpointSaver,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed import (
    DeepSpeedCheckpointer,
    StorageType,
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
        model_path = os.path.join(save_dir, str(tag), "model_states.pt")
        torch.save(model_sd, model_path)
        optimizer_sd = self.optimizer.state_dict()
        optim_path = os.path.join(save_dir, str(tag), "optim_states.pt")
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
        DeepSpeedCheckpointSaver._saver_instance = None
        DeepSpeedCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if DeepSpeedCheckpointSaver._saver_instance:
            DeepSpeedCheckpointSaver._saver_instance.close()
        clear_sock_dir()

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
            checkpointer = DeepSpeedCheckpointer(engine, tmpdirname)
            checkpointer.save_checkpoint(
                tmpdirname, step, storage_type=StorageType.MEMORY
            )
            shm_handler = checkpointer._async_save_engine._shm_handler
            self.assertFalse(shm_handler.no_checkpoint_state())
            self.assertEqual(
                checkpointer._async_save_engine._shm_handler._buffer_size, 9640
            )
            tensor_meta = (
                checkpointer._async_save_engine._shm_handler.metadata.get()
            )
            ds_ckpt_config = tensor_meta["_DLORVER_CKPT_CONFIG"]
            self.assertEqual(ds_ckpt_config.step, step)
            self.assertIsNotNone(tensor_meta["model_states"])
            tracer_file = os.path.join(tmpdirname, "latest")
            self.assertFalse(os.path.exists(tracer_file))

            checkpointer.save_checkpoint(
                tmpdirname, step, storage_type=StorageType.DISK
            )
            # Wait asynchronously saving.
            start = time.time()
            while True:
                if "100" in os.listdir(tmpdirname):
                    break
                time.sleep(0.1)
                if time.time() - start > 10:
                    break

            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
            self.assertTrue(os.path.exists(tracer_file))
            self.assertEqual(restored_step, step)

            files = sorted(os.listdir(tmpdirname + "/100"))
            self.assertEqual(files, ["model_states.pt", "optim_states.pt"])

            with self.assertRaises(ValueError):
                checkpointer.save_checkpoint(
                    tmpdirname, str(step), storage_type=2
                )

    def test_update_tracer_file(self):
        os.environ["LOCAL_RANK"] = "0"
        model = SimpleNet()
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.001,
        )
        step = 100
        engine = MockDeepSpeedEngine(model, optimizer)
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(os.path.join(tmpdirname, str(step)))
            checkpointer = DeepSpeedCheckpointer(engine, tmpdirname)
            dlrover_tracer_file = os.path.join(
                tmpdirname, CheckpointConstant.TRACER_FILE_NAME
            )
            with open(dlrover_tracer_file, "w") as f:
                f.write(str(50))
            checkpointer._update_tracer_file(step)
            ds_tracer_file = os.path.join(
                tmpdirname, DeepSpeedCheckpointSaver.TRACER_FILE
            )
            with open(ds_tracer_file, "r") as f:
                step = int(f.read())

            self.assertEqual(step, 50)
            os.remove(dlrover_tracer_file)
            checkpointer._update_tracer_file(step)
            self.assertFalse(os.path.exists(ds_tracer_file))
