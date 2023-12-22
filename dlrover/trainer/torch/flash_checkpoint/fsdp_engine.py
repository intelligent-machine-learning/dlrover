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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    LoadItemType,
    LoadPlan,
    LoadPlanner,
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

from dlrover.python.common.multi_process import SharedMemory
from dlrover.python.elastic_agent.torch.ckpt_saver import SharedMemoryHandler


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
        self.metadata: Dict[str, object] = {}

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
        if self.shm_handler.no_checkpint_state():
            buffer_size = _get_buffer_size(files, planner)
            self.shm_handler.init_shared_memory(create=True, size=buffer_size)
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
