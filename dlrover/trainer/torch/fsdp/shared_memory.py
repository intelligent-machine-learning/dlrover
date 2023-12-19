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
from typing import Dict, List, Tuple

import torch
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    Metadata,
    MetadataIndex,
    SavePlan,
    SavePlanner,
    StorageReader,
    StorageWriter,
    WriteItem,
    WriteItemType,
    WriteResult,
    _item_size,
    _result_from_write_item,
    _StorageInfo,
    _StoragePrefix,
    narrow_tensor_by_index,
)
from torch.futures import Future

from dlrover.python.common.multi_process import SharedMemory
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    SharedMemoryHandler,
    TensorMeta,
)


def _write_memory_from_list(
    shm: SharedMemory,
    files: List,
    planner: SavePlanner,
):
    for storage_key, write_items in files:
        write_results = []
        for write_item in write_items:
            data = planner.resolve_data(write_item)
            if torch.is_tensor(data):
                data = data.detach()
            write_results.append(
                _write_item(shm, data, write_item, storage_key)
            )
        return write_results


def _get_buffer_size(
    files: List[Tuple[str, List[WriteItem]]], planner: SavePlanner
):
    buffer_size = 0
    tensor_matadata = {}
    for _, write_items in files:
        for write_item in write_items:
            if write_item.type != WriteItemType.BYTE_IO:
                item_size = _item_size(write_item)
                buffer_size += item_size
            else:
                data = planner.resolve_data(write_item)
                tensor_matadata[write_item.index.fqn] = TensorMeta(
                    shape=data.shape,
                    dtype=data.dtype,
                    element_size=data.element_size(),
                    numel=data.numel(),
                    offset=buffer_size,
                )
                buffer_size += data.getbuffer().nbytes
    return buffer_size, tensor_matadata


def _write_item(shm: SharedMemory, data, write_item, storage_key):
    offset = 0
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

    return _result_from_write_item(
        write_item, length, _StorageInfo(storage_key, offset, length)
    )


class SharedMemoryWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        shm_handler: SharedMemoryHandler,
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
            shm_handler:
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
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> List[WriteResult]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        files = []
        self.file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
        for bucket in plan.items:
            files.append((self.file_name, bucket))
        if self.shm_handler.empty():
            buffer_size, tensor_metadata = _get_buffer_size(files, planner)
            self.metadata.update(tensor_metadata)
            self.shm_handler.init_tensor_shm(create=True, size=buffer_size)
        write_results = _write_memory_from_list(
            shm=self.shm_handler.shared_memory,
            files=files,
            planner=planner,
        )
        return write_results

    def finish(
        self, metadata: Metadata, results: List[List[WriteResult]]
    ) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        self.metadata["dcp_metadata"] = metadata


class SharedMemoryReader(StorageReader):
    def __init__(self, shm_handler: SharedMemoryHandler) -> None:
        super().__init__()
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.shm_handler = shm_handler

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            if read_item.type == LoadItemType.BYTE_IO:
                bytes = self.shm_handler.shared_memory.buf[
                    item_md.offset : item_md.length  # noqa E203
                ]
                planner.load_bytes(read_item, bytes)
            else:
                metadata = self.shm_handler.metadata.get()
                tensor_meta: TensorMeta = metadata[read_item.storage_index.fqn]
                tensor = torch.frombuffer(
                    buffer=self.shm_handler.shared_memory.buf,
                    dtype=tensor_meta.dtype,
                    offset=item_md.offset,
                    count=tensor_meta.numel,
                )
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
        return cached_meta["dcp_metadata"]

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool
    ) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(
        self, global_plan: List[LoadPlan]
    ) -> List[LoadPlan]:
        return global_plan
