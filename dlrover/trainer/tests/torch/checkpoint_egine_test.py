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
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlrover.python.common.constants import CheckpointConstant, NodeEnv
from dlrover.python.common.grpc import find_free_port
from dlrover.python.common.multi_process import clear_sock_dir
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    AsyncCheckpointSaver,
    CheckpointConfig,
    DeepSpeedCheckpointSaver,
    MegatronCheckpointSaver,
    TempDirCheckpointSaver,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed_engine import (
    DeepSpeedCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.engine import (
    check_all_rank_ready,
    start_saver_process,
    verify_all_rank_step_consistent,
)
from dlrover.trainer.torch.flash_checkpoint.full_ckpt_engine import (
    FullCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.megatron_engine import (
    MegatronCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.replica import (
    FullCkptReplicaManager,
)


def run_rank_sync(rank, ranks, world_size, master_port):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo")
    group = dist.new_group(ranks=ranks, backend="gloo")
    ready = check_all_rank_ready(group, True)
    passed = verify_all_rank_step_consistent(group, 10)
    dist.destroy_process_group()
    if not ready or not passed:
        raise ValueError("Async fails.")


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
    def persist_to_storage(
        self, local_shard_id, ckpt_config: CheckpointConfig
    ):
        state_dict = self._shm_handlers[local_shard_id].load_state_dict()
        for sd_name, sd in state_dict.items():
            if sd_name not in ckpt_config.paths:
                continue
            path = ckpt_config.paths[sd_name]
            torch.save(sd, path)

    def get_tracker_file(self):
        return os.path.join(self.checkpoint_dir, "tracker.txt")

    def update_tracker_file(self, step):
        with open(self.get_tracker_file(), "w") as f:
            f.write(str(step))


class SimpleShardingCheckpointEngine(FullCheckpointEngine):
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
        clear_sock_dir()

    def test_start_saver_proc(self):
        os.environ["ROLE_NAME"] = "dlrover-trainer"
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
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            saving_engine = SimpleShardingCheckpointEngine(tmpdir, storage)
            tmp = Path(tmpdir)
            saved_file = tmp / "checkpoint-100/checkpoint.pt"
            sd = {CheckpointConstant.MODEL_STATES_NAME: state_dict}
            paths = {CheckpointConstant.MODEL_STATES_NAME: saved_file}
            saving_engine.save_to_storage(step, sd, paths)
            time.sleep(3)
            # list the files in tmpdir recursively
            self.assertTrue(storage.exists(saved_file))

            tracker_file = tmp / "tracker.txt"
            self.assertTrue(storage.exists(tracker_file))

            self.assertEqual(tracker_file.read_text(), "100")
            state = torch.load(saved_file)
            self.assertEqual(state["step"], step)

            saver: AsyncCheckpointSaver = AsyncCheckpointSaver.get_ckpt_saver()
            saver.close()
            saving_engine.close()

    def test_megatron_engine(self):
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MegatronCheckpointEngine(tmpdir, storage)
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
            step, sd = engine.load()
            self.assertEqual(step, 0)
            self.assertDictEqual(sd, {})

    def test_deepspeed_engine(self):
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeepSpeedCheckpointEngine(
                tmpdir, storage, global_shard_num=1, zero_stage=1
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
            step = 100
            model_path = os.path.join(tmpdir, str(step), "model_states.pt")
            optim_path = os.path.join(tmpdir, str(step), "optim_states.pt")
            state_dict = {
                CheckpointConstant.MODEL_STATES_NAME: model.state_dict(),
                CheckpointConstant.OPTIM_STATES_NAME: optimizer.state_dict(),
            }
            paths = {
                CheckpointConstant.MODEL_STATES_NAME: model_path,
                CheckpointConstant.OPTIM_STATES_NAME: optim_path,
            }
            engine.save_to_storage(step, state_dict, paths)
            time.sleep(1)  # wait asynchronously saving
            self.assertEqual(engine._shm_handler._buffer_size, 9640)
            self.assertEqual(engine._shm_handler.shared_memory.size, 9640)
            restored_state_dict = engine.load()
            restore_msd = restored_state_dict[
                CheckpointConstant.MODEL_STATES_NAME
            ]
            msd = state_dict[CheckpointConstant.MODEL_STATES_NAME]
            for name in msd.keys():
                self.assertTrue(torch.equal(msd[name], restore_msd[name]))
            tracer_file = os.path.join(tmpdir, "latest")
            with open(tracer_file, "r") as f:
                restored_step = int(f.read())
                self.assertEqual(restored_step, step)

    @mock.patch("torch.distributed.barrier")
    def test_restore_memory_from_replica(self, mock_barrier):
        buffer = memoryview(b"123456789")
        meta = {"step": 100, "name": "test-weights"}
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            saving_engine = SimpleShardingCheckpointEngine(
                tmpdir, storage, replica_count=1
            )
            saving_engine._local_rank = 7
            with mock.patch.object(
                FullCkptReplicaManager,
                "gather",
                return_value=(torch.ByteTensor(buffer), meta),
            ):
                saving_engine._restore_memory_from_replica()
            shm_metadata = saving_engine._shm_handler.metadata.get()
            self.assertDictEqual(shm_metadata, meta)
            mock_barrier.assert_called()


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

        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdirname:
            engine = FullCheckpointEngine(tmpdirname, storage)
            engine._rank = 1
            path = os.path.join(tmpdirname, "10/rank_0.pt")
            ckpt_path = engine._gen_restore_checkpoint_path(10)
            self.assertEqual(path, ckpt_path)
            engine._restart_count = 1
            engine._notify_agent_to_create_saver()
            ckpt_dir = os.path.join(tmpdirname, "10")
            os.makedirs(ckpt_dir, exist_ok=True)
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

    def test_all_rank_sync(self):
        world_size = 2
        ranks = [i for i in range(world_size)]
        port = find_free_port()
        mp.spawn(
            run_rank_sync,
            nprocs=2,
            args=(ranks, world_size, port),
            join=True,
            daemon=False,
            start_method="spawn",
        )

    def test_sync_group(self):
        rank = 0
        world_size = 1
        master_port = find_free_port()
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(backend="gloo")
        try:
            storage = PosixDiskStorage()
            with tempfile.TemporaryDirectory() as tmpdirname:
                engine = FullCheckpointEngine(tmpdirname, storage)
                engine._init_sync_group("gloo")
                self.assertIsNone(engine._saver_group)
        finally:
            dist.destroy_process_group()


class PosixDiskStorageTest(unittest.TestCase):
    def test_posix(self):
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "test.txt")
            storage.write("100", tmp_path)
            content = storage.read(tmp_path)
            self.assertEqual(content, "100")
            storage.safe_remove(tmp_path)
            self.assertFalse(os.path.exists(tmp_path))
            test_dir = os.path.join(tmpdir, "test")
            storage.safe_makedirs(test_dir)
            self.assertTrue(os.path.exists(test_dir))
            storage.safe_rmtree(tmpdir)
            self.assertFalse(os.path.exists(test_dir))
            state_dict = {"weights": torch.rand(4, 4)}
            ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
            storage.write_state_dict(state_dict, ckpt_path, torch.save)
            storage.commit(100, True)
            sd = storage.read_state_dict(
                ckpt_path,
                read_func=lambda path: torch.load(path, map_location="cpu"),
            )
            self.assertTrue(torch.equal(state_dict["weights"], sd["weights"]))
