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
import unittest

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from dlrover.trainer.torch.elastic.checkpoint import (
    AsyncCheckpointEngine,
    CheckpointManger,
    get_latest_checkpoint,
)
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 60001)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNet()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.001,
        )
        self.dataset = SimpleDataset()
        self.sampler = ElasticDistributedSampler(
            dataset=self.dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            sampler=self.sampler,
        )

    def test_local_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_manager = CheckpointManger.init_checkpoint_manager(
                self.model,
                self.optimizer,
                self.dataloader,
                tmpdirname,
                max_to_keep=2,
            )
            ckpt_manager.save(epoch=0, step=10)
            ckpt_manager.save(epoch=0, step=20)
            ckpt_manager.save(epoch=0, step=30)
            ckpt_dirs = os.listdir(tmpdirname)
            self.assertEqual(len(ckpt_dirs), 2)

            ckpt_dir = get_latest_checkpoint(tmpdirname)
            expected_dir = os.path.join(tmpdirname, "checkpoint-30")
            self.assertEqual(ckpt_dir, expected_dir)

            ckpt_manager.load()
            self.assertEqual(self.dataloader.sampler.total_size, 60002)

    def test_ddp_save_load(self):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group(backend="gloo")
        model = DDP(self.model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_manager = CheckpointManger.init_checkpoint_manager(
                model,
                self.optimizer,
                self.dataloader,
                tmpdirname,
                max_to_keep=2,
            )
            ckpt_manager.save(epoch=0, step=10)
            ckpt_manager.save(epoch=0, step=20)
            ckpt_manager.save(epoch=0, step=30)
            ckpt_dirs = os.listdir(tmpdirname)
            self.assertEqual(len(ckpt_dirs), 2)

            ckpt_dir = get_latest_checkpoint(tmpdirname)
            expected_dir = os.path.join(tmpdirname, "checkpoint-30")
            self.assertEqual(ckpt_dir, expected_dir)

            ckpt_manager.load()
            self.assertEqual(self.dataloader.sampler.total_size, 60002)
        dist.destroy_process_group()
        print(dist.is_initialized())


class AsyncCheckpointEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.engine = AsyncCheckpointEngine(self.tmp_dir, 1, 1)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_memory(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        self.engine.save(step, state_dict)
        self.assertEqual(self.engine._shm_buffer[("step",)], 100)

        for key, value in state_dict["model"].items():
            buffer_value = self.engine._shm_buffer[("model", key)]
            self.assertTrue(value.equal(buffer_value))
