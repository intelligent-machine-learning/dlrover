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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from dlrover.python.common import grpc
from dlrover.python.elastic_agent.torch.ckpt_saver import CheckpointSaver
from dlrover.trainer.torch.elastic.checkpoint import (
    CheckpointManger,
    NoShardingCheckpointEngine,
    _get_latest_checkpoint,
)
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler


def set_torch_dist_env(port):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


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


def create_torch_modules():
    model = SimpleNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.001,
    )
    dataset = SimpleDataset()
    sampler = ElasticDistributedSampler(
        dataset=dataset,
        num_replicas=2,
        rank=0,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
    )
    return model, optimizer, dataloader


def _wait_async_saving_finished(dir_name, step):
    ckpt_path = os.path.join(dir_name, f"checkpoint-{step}/checkpoint.pt")
    while True:
        if os.path.exists(ckpt_path):
            return
        time.sleep(0.2)


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver.start_async_saving_ckpt()

    def test_ddp_save_load(self):
        os.environ["LOCAL_RANK"] = "0"
        port = grpc.find_free_port()
        set_torch_dist_env(port)
        dist.init_process_group(backend="gloo")
        model, optimizer, dataloader = create_torch_modules()
        model = DDP(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_manager = CheckpointManger.init_checkpoint_manager(
                model,
                optimizer,
                dataloader,
                tmpdirname,
                max_to_keep=2,
            )
            for step in [10, 20, 30]:
                ckpt_manager.save(epoch=0, step=step)
                _wait_async_saving_finished(tmpdirname, step)
            ckpt_dirs = os.listdir(tmpdirname)
            self.assertEqual(len(ckpt_dirs), 2)

            ckpt_dir = _get_latest_checkpoint(tmpdirname)
            expected_dir = os.path.join(tmpdirname, "checkpoint-30")
            self.assertEqual(ckpt_dir, expected_dir)

            ckpt_manager.load()
            self.assertEqual(dataloader.sampler.total_size, 60002)
            ckpt_manager._ckpt_engine.close()
        dist.destroy_process_group()

    def test_create_tensor_meta(self):
        engine = NoShardingCheckpointEngine("test-ckpt")
        value = torch.rand((10, 10), dtype=torch.float32)
        meta = engine._create_tensor_meta(value)
        self.assertEqual(meta.numel, 100)
        self.assertEqual(meta.element_size, 4)
        self.assertEqual(meta.offset, 0)
        self.assertEqual(meta.shape, (10, 10))
        self.assertEqual(meta.dtype, np.float32)
        engine.close()

    def test_load(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            engine = NoShardingCheckpointEngine(tmpdirname)
            path = os.path.join(tmpdirname, "checkpoint-10/checkpoint.pt")
            os.makedirs(os.path.dirname(path))
            torch.save(state_dict, path)
            path = os.path.join(tmpdirname, "checkpoint-20/checkpoint.pt")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("A error checkpoint\n")
            loaded_state_dict = engine._load_from_historic_checkpoint()
            for key, value in state_dict["model"].items():
                loaded_value = loaded_state_dict["model"][key]
                self.assertTrue(torch.equal(value, loaded_value))
            engine.close()
