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
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.elastic_agent.torch.ckpt_saver import CheckpointSaver
from dlrover.trainer.torch.elastic.checkpoint import CheckpointManger
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
    ckpt_path = os.path.join(dir_name, f"checkpoint-{step}.pt")
    while True:
        if os.path.exists(ckpt_path):
            return
        time.sleep(0.2)


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

    def test_ddp_save_load(self):
        os.environ["LOCAL_RANK"] = "0"
        port = grpc.find_free_port()
        set_torch_dist_env(port)
        dist.init_process_group(backend="gloo")
        try:
            model, optimizer, dataloader = create_torch_modules()
            model = DDP(model)
            msd = model.state_dict()
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
                ckpt_num = 0
                for d in ckpt_dirs:
                    if d.endswith(".pt"):
                        ckpt_num += 1
                self.assertEqual(ckpt_num, 2)

                tracer_file = os.path.join(
                    tmpdirname, CheckpointConstant.TRACER_FILE_NAME
                )
                with open(tracer_file, "r") as f:
                    restored_step = int(f.read())
                self.assertEqual(step, restored_step)

                ckpt_manager.load()
                self.assertEqual(dataloader.sampler.total_size, 60002)
                resume_msd = ckpt_manager.model.state_dict()
                self.assertTrue(
                    torch.equal(
                        msd["module.fc1.weight"],
                        resume_msd["module.fc1.weight"],
                    )
                )
                ckpt_manager._ckpt_engine.close()
        finally:
            dist.destroy_process_group()
