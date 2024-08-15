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
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlrover.python.common.constants import CheckpointConstant, NodeEnv
from dlrover.python.common.multi_process import (
    SharedDict,
    SharedMemory,
    SharedQueue,
)
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    AsyncCheckpointSaver,
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    CheckpointSharedObjPrefix,
    ClassMeta,
    DdpCheckpointSaver,
    FsdpDcpSaver,
    SharedMemoryHandler,
    _create_shared_memory,
    _traverse_state_dict,
)
from dlrover.python.tests.test_utils import start_local_master


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


class SharedMemoryHandlerTest(unittest.TestCase):
    def setUp(self):
        local_rank = 1
        os.environ[NodeEnv.TORCHELASTIC_RUN_ID] = "unittest"
        SharedDict(
            CheckpointSharedObjPrefix.META_NAME + str(local_rank), create=True
        )
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
        self._shm_handler.metadata.set({"step": 100})
        meta_dict = self._shm_handler.metadata.get()
        self.assertDictEqual(meta_dict, {"step": 100})


class CheckpointSaverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = PosixDiskStorage()
        AsyncCheckpointSaver._saver_instance = None
        AsyncCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if AsyncCheckpointSaver._saver_instance:
            AsyncCheckpointSaver._saver_instance.close()

    def test_create_checkpoint_saver(self):
        sq = SharedQueue(name="factory", create=False)
        class_meta = ClassMeta(
            module_path=DdpCheckpointSaver.__module__,
            class_name=DdpCheckpointSaver.__name__,
            kwargs={
                "checkpoint_dir": "test_ckpt",
                "storage_meta": self.storage.get_class_meta(),
            },
        )
        sq.put(class_meta)
        for _ in range(10):
            if AsyncCheckpointSaver._saver_instance is None:
                time.sleep(0.5)
            else:
                break
        self.assertIsNotNone(AsyncCheckpointSaver._saver_instance)
        AsyncCheckpointSaver.reset()
        wait = AsyncCheckpointSaver._saver_instance.wait_saving_checkpoint()
        self.assertFalse(wait)

        # test notify multiple times,
        # see if it will skip and no exception raised
        sq.put(class_meta)
        sq.put(class_meta)

        # test setup master client
        self.assertIsNone(AsyncCheckpointSaver._saver_instance._master_client)
        master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr)
        master_client = MasterClient.singleton_instance()
        self.assertIsNotNone(master_client)
        self.assertIsNotNone(
            AsyncCheckpointSaver._saver_instance.get_master_client()
        )
        self.assertIsNotNone(
            AsyncCheckpointSaver._saver_instance._master_client
        )
        self.assertEqual(
            id(master_client),
            id(AsyncCheckpointSaver._saver_instance._master_client),
        )

    def test_close_saver(self):
        saver = DdpCheckpointSaver("test_ckpt", self.storage.get_class_meta())
        try:
            SharedMemory(name="test").unlink()
        except Exception:
            pass
        saver._shm_handlers[0].shared_memory = SharedMemory(
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

    def test_create_shared_memory(self):
        shm = _create_shared_memory("test", False)
        self.assertIsNone(shm)

        shm = _create_shared_memory("test-repeat", True, size=10240)
        self.assertEqual(shm.size, 10240)

        shm = _create_shared_memory("test-repeat", True, size=102400)
        self.assertEqual(shm.size, 102400)
        shm.unlink()

    def test_save_to_storage(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DdpCheckpointSaver(tmpdir, self.storage.get_class_meta())
            path = Path(tmpdir) / "checkpoint.pt"
            paths = {CheckpointConstant.MODEL_STATES_NAME: path}
            ckpt_config = CheckpointConfig(step=100, paths=paths)
            state_dict = {
                CheckpointConstant.MODEL_STATES_NAME: state_dict,
                DLROVER_CKPT_CONFIG_KEY: ckpt_config,
            }
            saver._shm_handlers[0].save_state_dict(state_dict)
            meta_dict = saver._shm_handlers[0].metadata.get()
            ckpt_config: CheckpointConfig = meta_dict[DLROVER_CKPT_CONFIG_KEY]
            self.assertFalse(ckpt_config.writing_shm)
            self.assertEqual(ckpt_config.step, step)
            saver._shm_handlers[0].shared_memory = SharedMemory(
                name=saver._shm_handlers[0]._shm_name
            )
            AsyncCheckpointSaver._saver_instance = saver
            AsyncCheckpointSaver.register_signal_handler()
            handler = signal.getsignal(signal.SIGTERM)
            handler(None, None)
            with self.assertRaises(KeyboardInterrupt):
                handler = signal.getsignal(signal.SIGINT)
                handler(None, None)
            ckpt_files = os.listdir(tmpdir)
            self.assertEqual(len(ckpt_files), 3)
            saver.close()

            saver._node_rank = 1
            saver.persist_to_storage(0, None)

    def test_shard_num_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DdpCheckpointSaver(tmpdir, self.storage.get_class_meta())
            saver.global_shard_num = 1
            threading.Thread(
                target=saver._sync_shm_to_storage, daemon=True
            ).start()
            sq = SharedQueue(name="factory", create=True)
            saver._shm_handlers[0].init_shared_memory(create=True, size=1024)
            saver._shm_handlers[0].metadata.set({"step": 100})
            event = CheckpointEvent(
                type=CheckpointEventType.UPDATE_SHARD, global_shard_num=2
            )
            saver._event_queue.put(event)
            sq.unlink()
            time.sleep(0.3)
            self.assertEqual(saver.global_shard_num, 2)
            self.assertTrue(saver._shm_handlers[0].no_checkpoint_state())
            saver.close()

    def test_commit_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step_done_dir = os.path.join(tmpdir, ".done/10/")
            os.makedirs(step_done_dir, exist_ok=True)
            saver = DdpCheckpointSaver(tmpdir, self.storage.get_class_meta())
            saver.global_shard_num = 1
            saver.commit_checkpoint(100, step_done_dir, 2)

    def test_save_shm_to_storage(self):
        model = SimpleNet()
        step = 100
        state_dict = dict(
            model=model.state_dict(),
            step=step,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DdpCheckpointSaver(tmpdir, self.storage.get_class_meta())
            path = Path(tmpdir) / "checkpoint.pt"
            paths = {CheckpointConstant.MODEL_STATES_NAME: path}
            ckpt_config = CheckpointConfig(step=100, paths=paths)
            state_dict = {
                CheckpointConstant.MODEL_STATES_NAME: state_dict,
                DLROVER_CKPT_CONFIG_KEY: ckpt_config,
            }
            saver._shm_handlers[0].save_state_dict(state_dict)
            saver._writing_storage = True
            saver.save_shm_to_storage()
            self.assertFalse(saver._stop_commit)
            saver._sync_node_checkpoint = mock.MagicMock(return_value=False)
            saver.save_shm_to_storage(master_client=1)
            self.assertTrue(saver._stop_commit)
            saver.close()

    def test_report_failure(self):
        saver = DdpCheckpointSaver("test_ckpt", self.storage.get_class_meta())
        master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr)

        self.assertIsNone(saver._master_client)
        self.assertIsNotNone(saver.get_master_client())
        self.assertIsNotNone(saver._master_client)
        self.assertEqual(id(MasterClient._instance), id(saver._master_client))
        self.assertEqual(
            id(MasterClient._instance), id(saver.get_master_client())
        )
        saver._report_failure_to_master("test-error")


class FsdpCheckpointSaverTest(unittest.TestCase):
    def test_persist_storage(self):
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = FsdpDcpSaver(tmpdir, storage.get_class_meta())
            step = 100
            path = os.path.join(tmpdir, str(step), "__0_0.dist_cp")
            paths = {CheckpointConstant.MODEL_STATES_NAME: path}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ckpt_config = CheckpointConfig(
                step=step,
                writing_shm=False,
                paths=paths,
            )
            saver._shm_handlers[0].init_shared_memory(create=True, size=1024)
            dcp_metadata = {"weighits": 10}
            saver._shm_handlers[0].metadata.set({"dcp_metadata": dcp_metadata})
            saver._is_agent_rank_0 = True
            saver.persist_to_storage(0, ckpt_config)
            files = sorted(os.listdir(os.path.dirname(path)))
            self.assertListEqual(files, [".metadata", "__0_0.dist_cp"])
