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

import dataclasses
import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    StorageReader,
    StorageWriter,
    WriteItem,
    WriteItemType,
    WriteResult,
    narrow_tensor_by_index,
)
from torch.distributed.checkpoint.metadata import (
    STORAGE_TYPES,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.futures import Future

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import SharedMemory
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    FsdpDcpSaver,
    SharedMemoryHandler,
)

from .engine import (
    CheckpointEngine,
    check_all_rank_ready,
    timer,
    verify_all_rank_step_consistent,
)


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info.
    """

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


def _write_memory_from_list(
    shm: SharedMemory,
    files: List[Tuple[str, WriteItem]],
    planner: SavePlanner,
):
    write_results = []
    offset = 0
    no_shard_data: Dict[str, STORAGE_TYPES] = {}
    for storage_key, write_item in files:
        data = planner.resolve_data(write_item)
        if torch.is_tensor(data):
            data = data.detach()

        # Only rank-0 has the no sharding data and we need to broadcast
        # no sharding data to all ranks. All ranks can load the state dict
        # from the CPU memory.
        if write_item.type != WriteItemType.SHARD:
            no_shard_data[write_item.index.fqn] = data
        offset, write_result = _write_item(
            shm, offset, data, write_item, storage_key
        )
        write_results.append(write_result)
    return write_results, no_shard_data


def _tensor_item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _get_buffer_size(files: List[Tuple[str, WriteItem]], planner: SavePlanner):
    """Get the buffer size to storage all files."""
    buffer_size = 0
    for _, write_item in files:
        if write_item.type != WriteItemType.BYTE_IO:
            item_size = _tensor_item_size(write_item)
            buffer_size += item_size
        else:
            data = planner.resolve_data(write_item)
            buffer_size += data.getbuffer().nbytes
    return buffer_size


def _write_item(
    shm: SharedMemory, offset, data, write_item, storage_key
) -> Tuple[int, WriteResult]:
    """Write an item into the shared memory."""
    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        data_buf = data.getbuffer()
        length = data_buf.nbytes
        shm.buf[offset : offset + length] = data_buf  # noqa E203
    else:
        assert isinstance(data, torch.Tensor)
        shm_tensor = torch.frombuffer(
            shm.buf, dtype=data.dtype, count=data.numel(), offset=offset
        ).reshape(data.shape)
        shm_tensor.copy_(data)
        length = data.numel() * data.element_size()

    storage_data = _StorageInfo(storage_key, offset, length)
    write_result = WriteResult(
        index=write_item.index, size_in_bytes=length, storage_data=storage_data
    )
    offset += length
    return offset, write_result


class SharedMemoryWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using the shared memory.
    """

    def __init__(self, shm_handler: SharedMemoryHandler) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            shm_handler: A handler to write and read the shared memory.
        """
        super().__init__()
        self.file_name = ""
        self.shm_handler = shm_handler
        self.metadata: Dict[str, Any] = {}

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        return plan

    def prepare_global_plan(
        self, global_plan: List[SavePlan]
    ) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(global_plan)
        ]
        return new_plans

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        files = []
        self.file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
        for bucket in plan.items:
            files.append((self.file_name, bucket))
        if self.shm_handler.no_checkpoint_state():
            buffer_size = _get_buffer_size(files, planner)
            self.shm_handler.init_shared_memory(create=True, size=buffer_size)
        assert self.shm_handler.shared_memory is not None
        write_results, no_shard_data = _write_memory_from_list(
            shm=self.shm_handler.shared_memory,
            files=files,
            planner=planner,
        )
        self.metadata["no_shard_data"] = no_shard_data
        fut: Future[List[WriteResult]] = Future()
        fut.set_result(write_results)
        return fut

    def finish(
        self, metadata: Metadata, results: List[List[WriteResult]]
    ) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        self.metadata["dcp_metadata"] = metadata


class SharedMemoryReader(StorageReader):
    """
    Basic implementation of StorageReader using the shared memory.

    Args:
        shm_handler: A handler to write and read the shared memory.
    """

    def __init__(self, shm_handler: SharedMemoryHandler) -> None:
        super().__init__()
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.shm_handler = shm_handler
        self.state_dict_metadata: Dict[str, STORAGE_TYPES] = dict()
        self.no_shard_data: Dict[str, STORAGE_TYPES] = {}

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if self.shm_handler.shared_memory is None:
            self.shm_handler.init_shared_memory()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            pickled = False
            if not read_item.storage_index.offset:
                data = self.no_shard_data[read_item.storage_index.fqn]
                if isinstance(data, io.BytesIO):
                    bytes_io = data
                else:
                    bytes_io = io.BytesIO()
                    torch.save(data, bytes_io)
                bytes_io.seek(0)
                pickled = True
            else:
                end = item_md.offset + item_md.length
                assert self.shm_handler.shared_memory is not None
                bytes_view = self.shm_handler.shared_memory.buf[
                    item_md.offset : end  # noqa E203
                ]
                bytes_io = io.BytesIO(bytes_view)
                bytes_io.seek(0)

            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes_io)
            else:
                tensor_metadata: TensorStorageMetadata = (
                    self.state_dict_metadata[read_item.storage_index.fqn]
                )
                if pickled:
                    tensor = torch.load(bytes_io)
                else:
                    tensor_dtype = tensor_metadata.properties.dtype
                    tensor_shape = read_item.lengths

                    numel = 1
                    for i in tensor_shape:
                        numel = numel * i
                    tensor = torch.frombuffer(
                        buffer=bytes_io.getbuffer(),
                        dtype=tensor_dtype,
                    ).reshape(tensor_shape)
                tensor = narrow_tensor_by_index(
                    tensor, read_item.storage_offsets, read_item.lengths
                )
                target_tensor = planner.resolve_tensor(read_item).detach()

                err_msg = (
                    f"req {read_item.storage_index} mismatch sizes "
                    f"{target_tensor.size()} vs {tensor.size()}"
                )
                assert target_tensor.size() == tensor.size(), err_msg
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        cached_meta = self.shm_handler.metadata.get()
        dcp_metadata = cached_meta["dcp_metadata"]
        self.no_shard_data = cached_meta["no_shard_data"]
        return dcp_metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool
    ) -> None:
        self.storage_data = metadata.storage_data
        self.state_dict_metadata = metadata.state_dict_metadata
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(
        self, global_plan: List[LoadPlan]
    ) -> List[LoadPlan]:
        return global_plan


class SlicedBufferedReader(io.BufferedReader):
    def __init__(self, base_stream: io.RawIOBase, offset: int, len: int):
        super().__init__(base_stream)
        self.offset = offset
        self.len = len
        self.seek(0)

    def seek(self, __offset: int, __whence: int = os.SEEK_SET) -> int:
        if __whence == os.SEEK_SET:
            __offset = self.offset + __offset
        elif __whence == os.SEEK_END:
            __whence = os.SEEK_SET
            __offset = (self.offset + self.len) - __offset
        return super().seek(__offset, __whence)

    def tell(self) -> int:
        return super().tell() - self.offset


class FileReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = Path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.state_dict_metadata: Dict[str, STORAGE_TYPES] = dict()

    def _slice_file(self, file, sinfo: _StorageInfo):
        return SlicedBufferedReader(
            io.FileIO(file.fileno(), closefd=False), sinfo.offset, sinfo.length
        )

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    bytes = io.BytesIO(file_slice.read(item_md.length))
                    bytes.seek(0)
                    if req.type == LoadItemType.BYTE_IO:
                        planner.load_bytes(req, bytes)
                    else:
                        tensor_meta = self.state_dict_metadata[
                            req.storage_index.fqn
                        ]
                        tensor = torch.frombuffer(
                            bytes.getbuffer(),
                            dtype=tensor_meta.properties.dtype,
                        ).reshape(req.lengths)
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        err_msg = (
                            f"req {req.storage_index} mismatch sizes "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                        assert target_tensor.size() == tensor.size(), err_msg
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with (self.path / ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool
    ) -> None:
        self.storage_data = metadata.storage_data
        self.state_dict_metadata = metadata.state_dict_metadata
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(
        self, global_plan: List[LoadPlan]
    ) -> List[LoadPlan]:
        return global_plan


class FsdpCheckpointEngine(CheckpointEngine):
    """
    A engine to save FSDP distributed checkpoint into the memory
    and storage.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        storage,
        comm_backend="",
        save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    ):
        super().__init__(checkpoint_dir, storage, comm_backend, save_timeout)
        self._shm_writer = SharedMemoryWriter(shm_handler=self._shm_handler)
        self._shm_reader = SharedMemoryReader(self._shm_handler)

    def get_saving_ranks(self):
        """
        Only the local rank 0 in each node saves the state dict into the
        memory. They need to synchronize the saving status.
        """
        return None

    @timer
    def save_to_memory(self, step, state_dict, paths: Dict[str, str]):
        """
        Synchronously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the storage path to save the state dict.
                Note, the path is used to save the state dict to storage
                only if the training process fails.
        """
        if self._local_rank != self.local_shard_id:
            return False

        acquired = self._shm_lock.acquire(blocking=False)
        all_rank_ready = check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_lock.release()
            return False

        conf = CheckpointConfig(step=step)
        conf.writing_shm = True
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=self._shm_writer,
        )

        # Broadcast dcp metadata and no sharding data to all ranks
        # and all ranks can restore the state dict from the CPU
        # memory with those metadata.
        bcast_list = [self._shm_writer.metadata]
        dist.broadcast_object_list(bcast_list, src=0)
        self._shm_writer.metadata = bcast_list[0]

        model_sd_name = CheckpointConstant.MODEL_STATES_NAME
        path = os.path.join(paths[model_sd_name], self._shm_writer.file_name)
        conf.paths = {model_sd_name: path}
        meta_dict = {DLROVER_CKPT_CONFIG_KEY: conf}
        meta_dict.update(self._shm_writer.metadata)
        self._shm_handler.metadata.set(meta_dict)
        conf.writing_shm = False
        if acquired:
            self._shm_lock.release()
        self._cached_step = conf.step
        return True

    def save_to_storage(self, step, state_dict, paths: Dict[str, str]):
        """
        Save the state_dict into the path of storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the storage path to save the state dict.
        """
        succeed = True
        if step > self._cached_step:
            succeed = self.save_to_memory(step, state_dict, paths)

        if dist.is_initialized():
            dist.barrier()

        # Only local rank 0 on each node notifies the event to save.
        if self._local_rank == 0 and succeed:
            logger.info(
                "Put a save event to notify the agent persists checkpoint."
            )
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)

    def get_saver_class(self):
        """
        Get a CheckpointSaver class.
        """
        return FsdpDcpSaver

    def get_local_shard_num(self):
        """Get the number of model shards on the node."""
        return env_utils.get_local_world_size()

    def get_global_shard_num(self):
        """Get the number of model shards on all nodes."""
        return dist.get_world_size()

    def load(self, resume_path=""):
        """
        Get the checkpoint reader to read the state dict. The method
        firstly returns a shared memory reader if the shared memory
        has state dict. Otherwise, the method returns a file reader
        with the resume_path.

        Args:
            resume_path (str): the resuming path to load the
                checkpointing state dict from the storage.
        """
        default_config = CheckpointConfig()
        config = self._shm_handler.get_checkpoint_config(default_config)
        step = config.step
        passed = verify_all_rank_step_consistent(self._saver_group, step)
        if passed and not self._shm_handler.no_checkpoint_state():
            logger.info(f"Create a shared memory reader with step {step}.")
            return self._shm_reader
        else:
            if not resume_path:
                resume_path = self._get_track_resume_path()
            if os.path.exists(resume_path):
                f"Create a storage reader with path {resume_path}."
                return FileReader(resume_path)
            return None

    def _get_track_resume_path(self):
        tracker_file = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        step = self.storage.read(tracker_file)
        if step:
            return os.path.join(self.checkpoint_dir, step)
        else:
            return ""
