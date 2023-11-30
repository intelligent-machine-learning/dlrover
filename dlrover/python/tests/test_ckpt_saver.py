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
import torch.nn as nn
import torch.nn.functional as F

from dlrover.python.common.multi_process import SharedMemory, SharedQueue
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointSaver,
    NoShardingCheckpointEngine,
    NoShardingSaver,
    _clean_shm_handler,
    _convert_torch_dtype_to_numpy,
    _traverse_state_dict,
)


def set_torch_dist_env(port):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


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


class CheckpointSaverTest(unittest.TestCase):
    def test_create_checkpoint_saver(self):
        CheckpointSaver.start_async_saving_ckpt(num_proc=8)
        sq = SharedQueue(name="factory", create=False)
        args = {"checkpoint_dir": "test_dir"}
        sq.put(("NoShardingSaver", args))
        for _ in range(10):
            if CheckpointSaver._saver_instance is None:
                time.sleep(0.5)
            else:
                break
        self.assertEqual(CheckpointSaver._saver_instance.num_proc, 8)

    def test_close_saver(self):
        saver = NoShardingSaver("test_ckpt")
        saver._tensor_shm = SharedMemory(name="test", create=True, size=1024)
        saver.close()
        saver.close()

    def test_traverse_state_dict(self):
        def visitor(value):
            return value

        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        new_dict = _traverse_state_dict(state_dict, visitor)
        self.assertEqual(new_dict, state_dict)

    def test_convert_torch_dtype_to_numpy(self):
        np_dtype = _convert_torch_dtype_to_numpy(torch.float32)
        self.assertEqual(np_dtype, np.float32)

        np_dtype = _convert_torch_dtype_to_numpy(torch.float)
        self.assertEqual(np_dtype, np.float32)

        np_dtype = _convert_torch_dtype_to_numpy(torch.int32)
        self.assertEqual(np_dtype, np.int32)

    def test_save_to_storage(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            CheckpointSaver._saver_instance = NoShardingSaver(tmpdir)
            sq = SharedQueue(name="factory", create=True)
            saving_engine = NoShardingCheckpointEngine(tmpdir)
            saving_engine.save_to_memory(state_dict, step)
            saver: CheckpointSaver = CheckpointSaver.get_ckpt_saver()
            saver._tensor_shm = SharedMemory(name=saver._shm_name)
            saver.save_shm_to_storage()
            ckpt_files = os.listdir(tmpdir)
            self.assertEqual(len(ckpt_files), 1)
            sq.close()
            _clean_shm_handler(None, None)
