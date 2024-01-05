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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlrover.python.common.constants import CheckpointConstant, NodeEnv
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    AsyncCheckpointSaver,
    DeepSpeedCheckpointSaver,
    MegatronCheckpointSaver,
    TempDirCheckpointSaver,
)
from dlrover.trainer.torch.flash_checkpoint.ddp_engine import (
    DdpCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed_engine import (
    DeepSpeedCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.engine import start_saver_process
from dlrover.trainer.torch.flash_checkpoint.megatron_engine import (
    MegatronCheckpointEngine,
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


class SimpleShardingSaver(TempDirCheckpointSaver):
    def persist_to_storage(self, local_shard_id, path):
        state_dict = self._shm_handlers[local_shard_id].load_state_dict()
        state_file = os.path.join(path, "checkpoint.pt")
        torch.save(state_dict, state_file)

    def get_tracker_file(self):
        return os.path.join(self.checkpoint_dir, "tracker.txt")

    def update_tracker_file(self, step):
        with open(self.get_tracker_file(), "w") as f:
            f.write(str(step))


class SimpleShardingCheckpointEngine(DdpCheckpointEngine):
    def get_saver_class(self):
        return SimpleShardingSaver

    def get_global_shard_num(self):
        return 1

    def get_local_shard_num(self):
        return 1

    def load(self, resume_path=""):
        pass


class ShardingCheckpointEngineTest(unittest.TestCase):
    def setUp(self):
        os.environ[NodeEnv.NODE_NUM] = "1"
        os.environ[NodeEnv.NODE_RANK] = "0"
        AsyncCheckpointSaver._saver_instance = None
        AsyncCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self):
        os.environ.pop(NodeEnv.NODE_NUM, None)
        os.environ.pop(NodeEnv.NODE_RANK, None)
        if AsyncCheckpointSaver._saver_instance:
            AsyncCheckpointSaver._saver_instance.close()

    def test_start_saver_proc(self):
        proc = start_saver_process()
        self.assertIsNone(proc)
        os.environ["ROLE_NAME"] = "default"
        proc = start_saver_process()
        self.assertIsNotNone(proc)
        proc.kill()
        os.environ["ROLE_NAME"] = "dlrover-trainer"

    def test_save_to_storage(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saving_engine = SimpleShardingCheckpointEngine(tmpdir)
            saving_engine.save_to_storage(step, state_dict, "")
            tmp = Path(tmpdir)
            time.sleep(3)
            # list the files in tmpdir recursively
            saved_file = tmp / "checkpoint-100/checkpoint.pt"
            self.assertTrue(saved_file.exists())

            tracker_file = tmp / "tracker.txt"
            self.assertTrue(tracker_file.exists())

            self.assertEqual(tracker_file.read_text(), "100")
            state = torch.load(saved_file)
            self.assertEqual(state["step"], step)

            saver: AsyncCheckpointSaver = AsyncCheckpointSaver.get_ckpt_saver()
            saver.close()
            saving_engine.close()

    def test_megatron_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MegatronCheckpointEngine(tmpdir)
            global_shard_num = engine.get_global_shard_num()
            self.assertEqual(global_shard_num, 1)
            local_shard_num = engine.get_local_shard_num()
            self.assertEqual(local_shard_num, 1)

            saver_class = engine.get_saver_class()
            self.assertEqual(saver_class, MegatronCheckpointSaver)

            step = 100
            path = os.path.join(
                tmpdir, f"iter_0000{step}/mp_rank_00/model_optim_rng.pt"
            )

            tensor = torch.rand(4, 4)
            state_dict = {"weights": tensor}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

            tracker_file = os.path.join(
                tmpdir, CheckpointConstant.TRACER_FILE_NAME
            )
            with open(tracker_file, "w") as f:
                f.write(str(step))
            sd = engine._load_from_storage(path)
            self.assertTrue(torch.equal(sd["weights"], tensor))
            sd = engine.load(path)
            self.assertTrue(torch.equal(sd["weights"], tensor))

    def test_deepspeed_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeepSpeedCheckpointEngine(
                tmpdir, global_shard_num=1, zero_stage=1
            )
            global_shard_num = engine.get_global_shard_num()
            self.assertEqual(global_shard_num, 1)
            local_shard_num = engine.get_local_shard_num()
            self.assertEqual(local_shard_num, 1)

            saver_class = engine.get_saver_class()
            self.assertEqual(saver_class, DeepSpeedCheckpointSaver)
            model = SimpleNet()
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.001,
            )
            state_dict = dict(
                model_states=model.state_dict(),
                optim_states=optimizer.state_dict(),
            )
            step = 100
            model_path = os.path.join(tmpdir, str(step), "model_states.pt")
            optimizer_path = os.path.join(tmpdir, str(step), "optim_states.pt")
            engine.save_to_storage(
                step, state_dict, model_path, optimizer_path
            )
            time.sleep(1)  # wait asynchronouly saving
            self.assertEqual(engine._shm_handler._buffer_size, 9640)
            self.assertEqual(engine._shm_handler.shared_memory.size, 9640)
            restored_state_dict = engine.load()
            restore_msd = restored_state_dict["model_states"]
            msd = state_dict["model_states"]
            for name in msd.keys():
                self.assertTrue(torch.equal(msd[name], restore_msd[name]))
            tracer_file = os.path.join(tmpdir, "latest")
            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
                self.assertEqual(restored_step, step)


class CheckpointEngineTest(unittest.TestCase):
    def setUp(self):
        AsyncCheckpointSaver._saver_instance = None
        AsyncCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if AsyncCheckpointSaver._saver_instance:
            AsyncCheckpointSaver._saver_instance.close()

    def test_load_no_sharding(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            engine = DdpCheckpointEngine(tmpdirname)
            engine._restart_count = 1
            engine._notify_agent_to_create_saver()
            path = os.path.join(tmpdirname, "checkpoint-10.pt")
            torch.save(state_dict, path)
            tracer_file = os.path.join(
                tmpdirname, CheckpointConstant.TRACER_FILE_NAME
            )
            with open(tracer_file, "w") as f:
                f.write("10")

            loaded_state_dict = engine.load()
            for key, value in state_dict["model"].items():
                loaded_value = loaded_state_dict["model"][key]
                self.assertTrue(torch.equal(value, loaded_value))
            engine.close()

            with open(tracer_file, "w") as f:
                f.write("100")
            state_dict = engine._load_from_storage()
            self.assertDictEqual(state_dict, {})
