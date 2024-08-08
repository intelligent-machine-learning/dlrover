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

import io
import os
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ones
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    ChunkStorageMetadata,
    LoadItemType,
    LoadPlan,
    ReadItem,
    SavePlan,
    TensorProperties,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from dlrover.python.common import grpc
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.multi_process import SharedMemory, clear_sock_dir
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    AsyncCheckpointSaver,
    SharedMemoryHandler,
)
from dlrover.trainer.torch.flash_checkpoint.fsdp import FsdpShardCheckpointer
from dlrover.trainer.torch.flash_checkpoint.fsdp_engine import (
    FileReader,
    FsdpCheckpointEngine,
    SharedMemoryReader,
    SharedMemoryWriter,
    _get_buffer_size,
    _tensor_item_size,
    _write_item,
    _write_memory_from_list,
)

_OPTIMIZER_KEY = "optimizer.params.group"
_MODEL_TENSOR_KEY = "model.weights"


def set_torch_dist_env(port):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


def _make_tensor_write_item():
    index = MetadataIndex(
        _MODEL_TENSOR_KEY, offset=torch.Size([0, 0]), index=0
    )
    chunk = ChunkStorageMetadata(
        offsets=torch.Size([0, 0]), sizes=torch.Size([2, 4])
    )
    tensor_proper = TensorProperties(dtype=torch.float32)
    tensor_data = TensorWriteData(
        chunk=chunk, properties=tensor_proper, size=torch.Size([2, 4])
    )
    item = WriteItem(index, type=WriteItemType.TENSOR, tensor_data=tensor_data)
    return item


def _make_bytesio_write_item():
    index = MetadataIndex(_OPTIMIZER_KEY, offset=None, index=0)
    item = WriteItem(index, type=WriteItemType.BYTE_IO)
    return item


def _maker_state_dict_files():
    file_name = "__0_0.dist_cp"
    bytesio_item = _make_bytesio_write_item()
    tensor_item = _make_tensor_write_item()

    chunk_spec = ChunkShardingSpec(
        dim=0,
        placements=[
            "rank:0/cpu",
        ],
    )
    tensor = ones(chunk_spec, torch.Size([2, 4]))

    files = [(file_name, bytesio_item), (file_name, tensor_item)]
    state_dict = {
        _OPTIMIZER_KEY: {"learning_rate": 0.1},
        _MODEL_TENSOR_KEY: tensor,
    }
    return files, state_dict


def _write_state_dict_to_shm(shared_memory, files, state_dict):
    write_items: List[WriteItem] = []
    for _, item in files:
        write_items.append(item)

    save_plan = SavePlan(write_items)
    shm_handler = SharedMemoryHandler(0)
    shm_handler.shared_memory = shared_memory
    writer = SharedMemoryWriter(shm_handler)
    plans = writer.prepare_global_plan([save_plan])

    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, True)
    fut = writer.write_data(plans[0], planner)
    fut.wait()
    write_results = fut.value()

    chunk = ChunkStorageMetadata(
        offsets=torch.Size([0, 0]), sizes=torch.Size([2, 4])
    )
    tensor_storage_meta = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([2, 4]),
        chunks=[chunk],
    )
    bytes_storage_meta = BytesStorageMetadata()
    state_dict_metadata = {
        _OPTIMIZER_KEY: bytes_storage_meta,
        _MODEL_TENSOR_KEY: tensor_storage_meta,
    }

    metadata = Metadata(state_dict_metadata)
    writer.finish(metadata, [write_results])
    return writer


class FsdpCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.shm = SharedMemory(name="test_write_item", create=True, size=1024)
        AsyncCheckpointSaver._saver_instance = None
        AsyncCheckpointSaver.start_async_saving_ckpt()
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        port = grpc.find_free_port()
        set_torch_dist_env(port)
        dist.init_process_group(backend="gloo")

    def tearDown(self) -> None:
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("LOCAL_WORLD_SIZE", None)
        self.shm.unlink()
        dist.destroy_process_group()
        if AsyncCheckpointSaver._saver_instance:
            AsyncCheckpointSaver._saver_instance.close()
        clear_sock_dir()

    def test_tensor_item_size(self):
        item = _make_tensor_write_item()
        item_size = _tensor_item_size(item)
        self.assertEqual(item_size, 32)

    def test_write_item(self):
        data = torch.rand(2, 4)
        item = _make_tensor_write_item()
        file_name = "__0_0.dist_cp"
        offset, write_result = _write_item(
            self.shm,
            offset=0,
            data=data,
            write_item=item,
            storage_key=file_name,
        )
        self.assertEqual(offset, 32)
        self.assertEqual(write_result.index.offset, torch.Size([0, 0]))
        self.assertEqual(write_result.index.fqn, "model.weights")
        self.assertEqual(write_result.size_in_bytes, 32)
        self.assertEqual(write_result.storage_data.offset, 0)
        self.assertEqual(write_result.storage_data.length, 32)

        bytes_io = io.BytesIO()
        data = {"learning_rate": 0.1}
        torch.save(data, bytes_io)
        bytes_io.seek(0)
        item = _make_bytesio_write_item()
        offset, write_result = _write_item(
            self.shm,
            offset=offset,
            data=bytes_io,
            write_item=item,
            storage_key=file_name,
        )
        self.assertEqual(offset, 463)
        self.assertEqual(write_result.index.fqn, _OPTIMIZER_KEY)
        self.assertEqual(write_result.size_in_bytes, 431)
        self.assertEqual(write_result.storage_data.offset, 32)
        self.assertEqual(write_result.storage_data.length, 431)

    def test_write_memory_from_list(self):
        files, state_dict = _maker_state_dict_files()
        planner = DefaultSavePlanner()
        planner.set_up_planner(state_dict, True)
        write_results, no_shard_data = _write_memory_from_list(
            self.shm,
            files,
            planner,
        )
        self.assertTrue(_OPTIMIZER_KEY in no_shard_data)
        write_result = write_results[0]
        self.assertEqual(write_result.index.fqn, _OPTIMIZER_KEY)

    def test_get_buffer_size(self):
        files, state_dict = _maker_state_dict_files()
        planner = DefaultSavePlanner()
        planner.set_up_planner(state_dict, True)
        size = _get_buffer_size(files, planner)
        self.assertEqual(size, 463)

    def test_shared_memory_writer(self):
        files, state_dict = _maker_state_dict_files()
        write_items: List[WriteItem] = []
        for _, item in files:
            write_items.append(item)
        writer = _write_state_dict_to_shm(self.shm, files, state_dict)
        self.assertTrue("dcp_metadata" in writer.metadata)
        self.assertTrue("no_shard_data" in writer.metadata)

        writer.shm_handler.metadata.set(writer.metadata)
        reader = SharedMemoryReader(writer.shm_handler)
        dcp_metadata = reader.read_metadata()
        self.assertTrue(_OPTIMIZER_KEY in dcp_metadata.state_dict_metadata)
        self.assertTrue(_OPTIMIZER_KEY in reader.no_shard_data)
        reader.set_up_storage_reader(dcp_metadata, True)

        bytesio_item = write_items[0]
        read_bytesio_item = ReadItem(
            LoadItemType.BYTE_IO,
            bytesio_item.index,
            dest_offsets=None,
            storage_index=bytesio_item.index,
            storage_offsets=None,
            lengths=None,
        )
        tensor_item = write_items[1]
        read_tensor_item = ReadItem(
            LoadItemType.TENSOR,
            tensor_item.index,
            dest_offsets=torch.Size([0, 0]),
            storage_index=tensor_item.index,
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([2, 4]),
        )
        load_plan = LoadPlan([read_bytesio_item, read_tensor_item])
        reader.prepare_local_plan(load_plan)
        reader.prepare_global_plan([load_plan])

        load_planner = DefaultLoadPlanner()
        load_planner.set_up_planner(state_dict, dcp_metadata, True)
        reader.read_data(load_plan, load_planner)

    def test_file_reader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            files, state_dict = _maker_state_dict_files()
            write_items: List[WriteItem] = []
            for _, item in files:
                write_items.append(item)
            writer = _write_state_dict_to_shm(self.shm, files, state_dict)

            self.assertTrue("dcp_metadata" in writer.metadata)
            with (tmpdir / ".metadata").open("wb") as f:
                pickle.dump(writer.metadata["dcp_metadata"], f)
                os.fsync(f.fileno())
            with open(tmpdir / "__0_0.distcp", "wb") as f:
                f.write(writer.shm_handler.shared_memory.buf)

            reader = FileReader(tmpdir)
            metadata = reader.read_metadata()
            reader.set_up_storage_reader(metadata, True)

            bytesio_item = write_items[0]
            read_bytesio_item = ReadItem(
                LoadItemType.BYTE_IO,
                bytesio_item.index,
                dest_offsets=None,
                storage_index=bytesio_item.index,
                storage_offsets=None,
                lengths=None,
            )
            tensor_item = write_items[1]
            read_tensor_item = ReadItem(
                LoadItemType.TENSOR,
                tensor_item.index,
                dest_offsets=torch.Size([0, 0]),
                storage_index=tensor_item.index,
                storage_offsets=torch.Size([0, 0]),
                lengths=torch.Size([2, 4]),
            )
            load_plan = LoadPlan([read_bytesio_item, read_tensor_item])

            reader.prepare_local_plan(load_plan)
            reader.prepare_global_plan([load_plan])

            load_planner = DefaultLoadPlanner()
            load_planner.set_up_planner(state_dict, metadata, True)
            reader.read_data(load_plan, load_planner)

    def test_fsdp_engine(self):
        step = 100
        state_dict = {
            _OPTIMIZER_KEY: {"learning_rate": 0.1},
        }
        storage = PosixDiskStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            path = tmpdir / str(step)
            paths = {CheckpointConstant.MODEL_STATES_NAME: path}
            engine = FsdpCheckpointEngine(tmpdir, storage)
            engine.save_to_storage(
                step,
                state_dict,
                paths=paths,
            )
            self.assertEqual(engine._cached_step, 100)
            time.sleep(1)
            files = sorted(os.listdir(tmpdir))
            self.assertListEqual(
                files,
                [
                    "._dlrover_ckpt_stage",
                    "100",
                    "dlrover_latest.txt",
                ],
            )
            files = sorted(os.listdir(path))
            self.assertListEqual(files, [".metadata", "__0_0.distcp"])
            reader = engine.load(path)
            self.assertTrue(isinstance(reader, SharedMemoryReader))

            path = engine._get_track_resume_path()
            self.assertEqual(path, os.path.join(tmpdir, "100"))
            tracker_file = os.path.join(
                tmpdir, CheckpointConstant.TRACER_FILE_NAME
            )
            with open(tracker_file, "w") as f:
                f.write("")
            path = engine._get_track_resume_path()
            self.assertEqual(path, "")
            os.remove(tracker_file)
            path = engine._get_track_resume_path()
            self.assertEqual(path, "")

    def test_fsdp_checkpointer(self):
        step = 100
        state_dict = {
            _OPTIMIZER_KEY: {"learning_rate": 0.1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpointer = FsdpShardCheckpointer(tmpdir)
            path = tmpdir / str(step)
            engine = checkpointer._engine
            paths = {CheckpointConstant.MODEL_STATES_NAME: path}
            checkpointer._engine.save_to_storage(step, state_dict, paths)
            self.assertEqual(engine._cached_step, 100)
            time.sleep(1)
            files = sorted(os.listdir(tmpdir))
            self.assertListEqual(
                files,
                [
                    "._dlrover_ckpt_stage",
                    "100",
                    "dlrover_latest.txt",
                ],
            )
            files = sorted(os.listdir(path))
            self.assertListEqual(files, [".metadata", "__0_0.distcp"])
            reader = checkpointer._engine.load(path)
            self.assertTrue(isinstance(reader, SharedMemoryReader))
