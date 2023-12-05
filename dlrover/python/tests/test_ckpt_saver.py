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
import signal
import tempfile
import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlrover.python.common.multi_process import SharedMemory, SharedQueue
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    _WIRTING_SHM,
    CheckpointSaver,
    NoShardingCheckpointEngine,
    NoShardingSaver,
    SaverClassMeta,
    SharedMemoryHandler,
    _create_shared_memory,
    _load_from_historic_checkpoint,
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
        CheckpointSaver.start_async_saving_ckpt()
        sq = SharedQueue(name="factory", create=False)
        class_meta = SaverClassMeta(
            module_path="dlrover.python.elastic_agent.torch.ckpt_saver",
            class_name="NoShardingSaver",
            init_args={"checkpoint_dir": "test_ckpt"},
        )
        sq.put(class_meta)
        for _ in range(10):
            if CheckpointSaver._saver_instance is None:
                time.sleep(0.5)
            else:
                break

    def test_close_saver(self):
        saver = NoShardingSaver("test_ckpt")
        saver._shm_handler._tensor_shm = SharedMemory(
            name="test",
            create=True,
            size=1024,
        )
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
            meta_dict = saving_engine._shm_handler._tensor_meta._dict
            self.assertFalse(meta_dict[_WIRTING_SHM])
            saver: NoShardingSaver = CheckpointSaver.get_ckpt_saver()
            saver._shm_handler._tensor_shm = SharedMemory(
                name=saver._shm_handler._shm_name
            )
            CheckpointSaver.register_signal_handler()
            handler = signal.getsignal(signal.SIGTERM)
            handler(None, None)
            with self.assertRaises(KeyboardInterrupt):
                handler = signal.getsignal(signal.SIGINT)
                handler(None, None)
            ckpt_files = os.listdir(tmpdir)
            self.assertEqual(len(ckpt_files), 1)
            sq.unlink()


class CheckpointEngineTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def test_create_shared_memory(self):
        shm = _create_shared_memory("test", False)
        self.assertIsNone(shm)

    def test_create_tensor_meta(self):
        shm_handler = SharedMemoryHandler(0, host=False)
        value = torch.rand((10, 10), dtype=torch.float32)
        meta = shm_handler._create_tensor_meta(value)
        self.assertEqual(meta.numel, 100)
        self.assertEqual(meta.element_size, 4)
        self.assertEqual(meta.offset, 0)
        self.assertEqual(meta.shape, (10, 10))
        self.assertEqual(meta.dtype, torch.float32)
        shm_handler.close()

    def test_load_no_sharding(self):
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
            loaded_state_dict = _load_from_historic_checkpoint(
                engine.checkpoint_dir
            )
            for key, value in state_dict["model"].items():
                loaded_value = loaded_state_dict["model"][key]
                self.assertTrue(torch.equal(value, loaded_value))
            engine.close()
