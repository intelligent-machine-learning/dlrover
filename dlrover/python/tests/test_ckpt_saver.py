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
import threading
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlrover.python.common.constants import CheckpointConstant, NodeEnv
from dlrover.python.common.multi_process import (
    SharedDict,
    SharedMemory,
    SharedQueue,
)
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    _CKPT_META_NAME_PREFIX,
    _DLROVER_CKPT_KEY,
    AsyncCheckpointSaver,
    AtorchFSDPShardingSaver,
    CheckpointSaver,
    CheckpointShardConfig,
    FSDPShardingCheckpointEngine,
    MegatronCheckpointEngine,
    NoShardingCheckpointEngine,
    SaverClassMeta,
    SharedMemoryHandler,
    _create_shared_memory,
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


class ShardingEngineDemo(NoShardingCheckpointEngine):
    def __init__(self, checkpoint_dir, global_shard_num=1):
        self._global_shard_num = global_shard_num
        super().__init__(checkpoint_dir)

    def get_global_shard_num(self):
        return self._global_shard_num


class SharedMemoryHandlerTest(unittest.TestCase):
    def setUp(self):
        local_rank = 1
        SharedDict(_CKPT_META_NAME_PREFIX + str(local_rank), create=True)
        self._shm_handler = SharedMemoryHandler(local_rank, host=False)

    def tearDown(self):
        self._shm_handler.close()

    def test_create_tensor_meta(self):
        value = torch.rand((10, 10), dtype=torch.float32)
        meta = self._shm_handler._create_tensor_meta(value)
        self.assertEqual(meta.numel, 100)
        self.assertEqual(meta.element_size, 4)
        self.assertEqual(meta.offset, 0)
        self.assertEqual(meta.shape, (10, 10))
        self.assertEqual(meta.dtype, torch.float32)

    def test_load_state_dict(self):
        state_dict = self._shm_handler.load_state_dict()
        self.assertDictEqual(state_dict, {})
        self._shm_handler._tensor_meta.set({"step": 100})
        meta_dict = self._shm_handler._tensor_meta.get()
        self.assertDictEqual(meta_dict, {"step": 100})


class CheckpointSaverTest(unittest.TestCase):
    def setUp(self) -> None:
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

    def test_create_checkpoint_saver(self):
        sq = SharedQueue(name="factory", create=False)
        class_meta = SaverClassMeta(
            module_path=AsyncCheckpointSaver.__module__,
            class_name=AsyncCheckpointSaver.__name__,
            init_args={"checkpoint_dir": "test_ckpt"},
        )
        sq.put(class_meta)
        for _ in range(10):
            if CheckpointSaver._saver_instance is None:
                time.sleep(0.5)
            else:
                break
        self.assertIsNotNone(CheckpointSaver._saver_instance)

    def test_close_saver(self):
        saver = AsyncCheckpointSaver("test_ckpt")
        try:
            SharedMemory(name="test").unlink()
        except Exception:
            pass
        saver._shm_handlers[0]._tensor_shm = SharedMemory(
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
            CheckpointSaver._saver_instance = AsyncCheckpointSaver(tmpdir)
            sq = SharedQueue(name="factory", create=True)
            saving_engine = NoShardingCheckpointEngine(tmpdir)
            sq.unlink()
            saving_engine.save_to_memory(step, state_dict)
            meta_dict = saving_engine._shm_handler._tensor_meta._dict
            ckpt_config: CheckpointShardConfig = meta_dict[_DLROVER_CKPT_KEY]
            self.assertFalse(ckpt_config.writing_shm)
            self.assertEqual(ckpt_config.step, step)
            saver: AsyncCheckpointSaver = CheckpointSaver.get_ckpt_saver()
            saver._shm_handlers[0]._tensor_shm = SharedMemory(
                name=saver._shm_handlers[0]._shm_name
            )
            saver._writing_storage = True
            saver.save_shm_to_storage(timeout=2)
            saver._writing_storage = False
            conf = saving_engine._shm_handler.get_checkpoint_config()
            self.assertEqual(conf.step, step)
            CheckpointSaver.register_signal_handler()
            handler = signal.getsignal(signal.SIGTERM)
            handler(None, None)
            with self.assertRaises(KeyboardInterrupt):
                handler = signal.getsignal(signal.SIGINT)
                handler(None, None)
            ckpt_files = os.listdir(tmpdir)
            self.assertEqual(len(ckpt_files), 1)
            saver.close()

    def test_shard_num_changes(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = AsyncCheckpointSaver(tmpdir)
            threading.Thread(
                target=saver._sync_shm_to_storage, daemon=True
            ).start()
            # Mock a shared queue for the engine.
            sq = SharedQueue(name="factory", create=True)
            saving_engine = ShardingEngineDemo(tmpdir, 1)
            sq.unlink()
            saving_engine.save_to_memory(step, state_dict)
            sq = SharedQueue(name="factory", create=True)
            saving_engine = ShardingEngineDemo(tmpdir, 2)
            sq.unlink()
            self.assertTrue(saver._shm_handlers[0].empty())
            self.assertIsNone(saver._shm_handlers[0]._tensor_shm)
            saving_engine.save_to_memory(step, state_dict)
            self.assertFalse(saver._shm_handlers[0].empty())
            saver.close()

    def test_commit_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step_done_dir = os.path.join(tmpdir, ".done/10/")
            os.makedirs(step_done_dir, exist_ok=True)
            saver = AsyncCheckpointSaver(tmpdir)
            saver.global_shard_num = 1
            saver.commit_checkpoint(100, step_done_dir, 2)


class CheckpointEngineTest(unittest.TestCase):
    def setUp(self):
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

    def test_create_shared_memory(self):
        shm = _create_shared_memory("test", False)
        self.assertIsNone(shm)

        shm = _create_shared_memory("test-repeat", True, size=10240)
        self.assertEqual(shm.size, 10240)

        shm = _create_shared_memory("test-repeat", True, size=102400)
        self.assertEqual(shm.size, 102400)

    def test_load_no_sharding(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            engine = NoShardingCheckpointEngine(tmpdirname)
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


class SimpleShardingSaver(AtorchFSDPShardingSaver):
    def persist_to_storage(self, state_dict, path):
        state_file = os.path.join(path, "checkpoint.pt")
        torch.save(state_dict, state_file)

    def get_tracker_file(self):
        return os.path.join(self.checkpoint_dir, "tracker.txt")

    def update_tracker_file(self, step):
        with open(self.get_tracker_file(), "w") as f:
            f.write(str(step))


class SimpleShardingCheckpointEngine(FSDPShardingCheckpointEngine):
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
        CheckpointSaver._saver_instance = None
        CheckpointSaver.start_async_saving_ckpt()

    def tearDown(self):
        os.environ.pop(NodeEnv.NODE_NUM, None)
        os.environ.pop(NodeEnv.NODE_RANK, None)
        if CheckpointSaver._saver_instance:
            CheckpointSaver._saver_instance.close()

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

            saver: CheckpointSaver = CheckpointSaver.get_ckpt_saver()
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
            self.assertEqual(saver_class, AsyncCheckpointSaver)

            step = 100
            path = engine._get_checkpoint_name(step)
            expected_path = os.path.join(
                tmpdir, "iter_0000100/mp_rank_00/model_optim_rng.pt"
            )
            self.assertEqual(path, expected_path)

            tensor = torch.rand(4, 4)
            state_dict = {"weights": tensor}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

            sd = engine._load_from_storage()
            self.assertDictEqual(sd, {})

            tracker_file = (
                AsyncCheckpointSaver.get_checkpoint_tracker_filename(tmpdir)
            )
            with open(tracker_file, "w") as f:
                f.write(str(step))
            sd = engine._load_from_storage(path)
            self.assertTrue(torch.equal(sd["weights"], tensor))
            sd = engine._load_from_storage()
            self.assertTrue(torch.equal(sd["weights"], tensor))
            sd = engine.load()
            self.assertTrue(torch.equal(sd["weights"], tensor))
