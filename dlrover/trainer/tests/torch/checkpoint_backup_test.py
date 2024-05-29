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
import shutil
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from dlrover.python.common.multi_process import SOCKET_TMP_DIR
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    CheckpointConfig,
    SharedMemoryHandler,
)
from dlrover.trainer.torch.flash_checkpoint.ckpt_backup import (
    FullCkptBackupManager,
    ShardCkptBackupManager,
)

CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def cleanup():
    dist.destroy_process_group()


def run_checkpoint_backup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    shm_hanlder = SharedMemoryHandler(local_rank=rank)
    config = CheckpointConfig(rank=rank, world_size=world_size)
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        DLROVER_CKPT_CONFIG_KEY: config,
    }
    shm_hanlder.save_state_dict(state_dict)

    with mock.patch.object(
        ShardCkptBackupManager, "_get_backup_ranks", return_value=[0, 1]
    ):
        back_manager = ShardCkptBackupManager(
            local_rank=rank, local_world_size=1, backup_group_size=2
        )
    back_manager.backup_ranks = list(range(world_size))
    back_manager.backup(shm_hanlder)
    if rank == 0:
        shm_hanlders = [shm_hanlder, shm_hanlder]
    else:
        peer_shm_handler = SharedMemoryHandler(2)
        shm_hanlders = [peer_shm_handler, peer_shm_handler]
    shm_tensor, _ = back_manager._gather_owner_checkpoint(shm_hanlders)
    if rank == 0 and shm_tensor.numel() != 1632:
        raise ValueError("Test Failed!")

    with mock.patch.object(
        FullCkptBackupManager, "_get_backup_ranks", return_value=[0, 1]
    ):
        back_manager = FullCkptBackupManager(
            local_rank=rank, local_world_size=1
        )
    shm_tensor, _ = back_manager._gather_owner_checkpoint(shm_hanlder)
    if rank == 0 and shm_tensor.numel() != 1632:
        raise ValueError("Test Failed!")
    cleanup()


class CheckpointBackupTest(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(SOCKET_TMP_DIR, ignore_errors=True)

    @mock.patch("torch.distributed.new_group")
    @mock.patch("torch.distributed.get_rank")
    def test_get_backup_ranks(self, mock_new_group, mock_get_rank):
        mock_get_rank.return_value = 1
        shard_manager = ShardCkptBackupManager(0, 8)
        self.assertListEqual(shard_manager.backup_ranks, [0, 8])

        os.environ["NODE_NUM"] = "4"
        shard_manager = FullCkptBackupManager(0, 8)
        self.assertListEqual(shard_manager.backup_ranks, [0, 8, 16, 24])

    def test_backup_checkpoint(self):
        world_size = 2
        mp.spawn(
            run_checkpoint_backup,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
