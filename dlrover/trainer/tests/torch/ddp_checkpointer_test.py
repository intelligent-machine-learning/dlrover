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

import tempfile
import unittest

import torch.nn as nn
import torch.nn.functional as F

from dlrover.python.common.multi_process import clear_sock_dir
from dlrover.python.common.storage import (
    KeepLatestStepStrategy,
    PosixStorageWithDeletion,
)
from dlrover.python.elastic_agent.torch.ckpt_saver import DdpCheckpointSaver
from dlrover.trainer.torch.flash_checkpoint.ddp import (
    DdpCheckpointer,
    StorageType,
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


class DdpCheckpoinerTest(unittest.TestCase):
    def setUp(self) -> None:
        DdpCheckpointSaver._saver_instance = None
        DdpCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if DdpCheckpointSaver._saver_instance:
            DdpCheckpointSaver._saver_instance.close()
        clear_sock_dir()

    def test_ddp_checkpointer(self):
        model = SimpleNet()
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = KeepLatestStepStrategy(
                max_to_keep=1, checkpoint_dir=tmpdir
            )
            checkpointer = DdpCheckpointer(tmpdir, deletion_strategy=strategy)
            self.assertTrue(
                isinstance(checkpointer.storage, PosixStorageWithDeletion)
            )
            step = 100
            sd = {"model": model.state_dict()}
            checkpointer.save_checkpoint(
                step, sd, storage_type=StorageType.MEMORY
            )
            sd = checkpointer.load_checkpoint()
            self.assertTrue("model" in sd)
