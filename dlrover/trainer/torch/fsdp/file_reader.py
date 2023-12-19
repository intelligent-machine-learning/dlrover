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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.distributed.checkpoint.filesystem import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    Metadata,
    MetadataIndex,
    ReadItem,
    StorageReader,
    narrow_tensor_by_index,
)
from torch.distributed.checkpoint.metadata import STORAGE_TYPES
from torch.futures import Future


@dataclass
class StorageInfo:
    """
    This is the per entry storage info
    """

    relative_path: str
    offset: int
    length: int


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
        self.storage_data: Dict[MetadataIndex, StorageInfo] = dict()
        self.state_dict_metadata: Dict[str, STORAGE_TYPES] = dict()

    def _slice_file(self, file, sinfo: StorageInfo):
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
                # TODO sort by offset and cache the reading
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
                            bytes, dtype=tensor_meta.properties.dtype
                        )
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
