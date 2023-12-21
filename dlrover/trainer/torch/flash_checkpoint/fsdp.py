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


from .checkpointer import Checkpointer, StorageType
from .fsdp_engine import FsdpCheckpointEngine


class FsdpCheckpointer(Checkpointer):
    def __init__(self, checkpoint_dir: str):
        self._engine = FsdpCheckpointEngine(checkpoint_dir)

    def save_checkpoint(
        self, step, state_dict, path, storage_type=StorageType.DISK
    ):
        if storage_type == StorageType.DISK and not path:
            raise ValueError("path cannot be empty if storage type is disk!")
        if storage_type == StorageType.MEMORY:
            self._engine.save_to_memory(step, state_dict, path)
        elif storage_type == StorageType.DISK:
            self._engine.save_to_storage(step, state_dict, path)
        else:
            raise ValueError(f"No support storage type {storage_type}")

    def load_checkpoint(self, resume_path=""):
        return self._engine.load(resume_path)
