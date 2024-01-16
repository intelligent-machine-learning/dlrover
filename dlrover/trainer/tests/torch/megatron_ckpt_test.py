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
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    MegatronCheckpointSaver,
)
from dlrover.trainer.torch.flash_checkpoint import megatron
from dlrover.trainer.torch.flash_checkpoint.checkpointer import StorageType
from dlrover.trainer.torch.flash_checkpoint.megatron import (
    MegatronCheckpointManager,
    load_checkpoint,
    save_checkpoint,
)


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
        MegatronCheckpointSaver._saver_instance = None
        MegatronCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if MegatronCheckpointSaver._saver_instance:
            MegatronCheckpointSaver._saver_instance.close()

    def test_save_load(self):
        os.environ["LOCAL_RANK"] = "0"
        model = SimpleNet()
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.001,
        )
        with tempfile.TemporaryDirectory() as tmpdirname:

            suffix = "model_optim_rng.pt"

            def get_args():
                parser = argparse.ArgumentParser(description="Megatron Test")
                args, _ = parser.parse_known_args()
                args.save = tempfile
                return args

            def mock_save_checkpoint(
                iteration, model, optimizer, opt_scheduler
            ):
                state_dict = {
                    "iteration": iteration,
                    "model_states": model.state_dict(),
                    "optim_states": optimizer.state_dict(),
                }
                path = os.path.join(tmpdirname, str(iteration), suffix)
                torch.save(state_dict, path)

            def mock_load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                load_arg="load",
                strict=True,
            ):
                path = os.path.join(tmpdirname, "20", suffix)
                state_dict = torch.load(path)
                model.load_state_dict(state_dict["model_states"])
                optimizer.load_state_dict(state_dict["optim_states"])

            megatron.megatron_save = mock_save_checkpoint
            megatron.megatron_load = mock_load_checkpoint
            megatron.get_args = get_args

            ckpt_manager = MegatronCheckpointManager(tmpdirname)
            save_checkpoint(
                10, model, optimizer, None, storage_type=StorageType.MEMORY
            )
            self.assertFalse(
                ckpt_manager.engine._shm_handler.no_checkpint_state()
            )
            self.assertEqual(
                ckpt_manager.engine._shm_handler._buffer_size, 9640
            )
            meta_dict = ckpt_manager.engine._shm_handler.metadata.get()
            meta_dict = meta_dict[CheckpointConstant.MODEL_STATES_NAME]
            self.assertEqual(meta_dict["iteration"], 10)
            self.assertIsNotNone(meta_dict["model_states"])

            save_checkpoint(
                20, model, optimizer, None, storage_type=StorageType.DISK
            )

            # Wait asynchronously saving.
            while True:
                if "20" in os.listdir(tmpdirname):
                    break
                time.sleep(0.1)

            tracer_file = os.path.join(
                tmpdirname, MegatronCheckpointSaver.TRACER_FILE
            )
            self.assertTrue(os.path.exists(tracer_file))
            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
            self.assertEqual(restored_step, 20)
            files = os.listdir(tmpdirname + "/20")
            self.assertEqual(files, [suffix])
            load_checkpoint(model, optimizer, None)

            with self.assertRaises(ValueError):
                save_checkpoint(20, model, optimizer, None, storage_type=2)

    def test_update_tracer_file(self):
        os.environ["LOCAL_RANK"] = "0"
        iteration = 100
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_dir = os.path.join(
                tmpdirname, "iter_{:07d}".format(iteration)
            )
            os.makedirs(ckpt_dir)
            ckpt_manager = MegatronCheckpointManager(tmpdirname)
            ckpt_manager.checkpoint_dir = tmpdirname
            dlrover_tracer_file = os.path.join(
                tmpdirname, CheckpointConstant.TRACER_FILE_NAME
            )
            with open(dlrover_tracer_file, "w") as f:
                f.write(str(50))
            ckpt_manager.update_tracer_file(iteration)
            self.assertFalse(os.path.exists(ckpt_dir))
            megatron_tracer_file = os.path.join(
                tmpdirname, MegatronCheckpointSaver.TRACER_FILE
            )
            with open(megatron_tracer_file, "r") as f:
                step = int(f.read())

            self.assertEqual(step, 50)
            os.remove(dlrover_tracer_file)
            ckpt_manager.update_tracer_file(step)
            self.assertFalse(os.path.exists(megatron_tracer_file))
