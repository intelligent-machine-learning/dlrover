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

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.storage import PosixDiskStorage

from .checkpointer import Checkpointer, StorageType
from .fsdp_engine import FsdpCheckpointEngine


class FsdpCheckpointer(Checkpointer):
    """
    Flash checkpointer saves and loads a FSDP module.

    Args:
        checkpoint_dir: the directory to save the checkpoint.
        storage: A CheckpointStorage instance. The checkpointer will
            use a PosixStorage instance if the storage is not defined.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.

    Examples::
        >>> checkpointer = FsdpCheckpointer(checkpoint_dir)
        >>> # Save checkpoint
        >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        >>>     state_dict = {
        >>>         "model": model.state_dict(),
        >>>         "optim": FSDP.optim_state_dict(model, optimizer),
        >>>     }
        >>> ckpt_dir = os.path.join(checkpoint_dir, str(step))
        >>> if step % save_memory_interval == 0:
        >>>     checkpointer.save_checkpoint(
        >>>         step, state_dict, ckpt_dir, storage_type=StorageType.MEMORY
        >>>     )
        >>> if step % save_storage_interval == 0:
        >>>     checkpointer.save_checkpoint(
        >>>         step, state_dict, ckpt_dir, storage_type=StorageType.DISK
        >>>     )
        >>> # Load checkpoint
        >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        >>> state_dict = {"model": model.state_dict()}
        >>> storage_reader = checkpointer.get_storage_reader()
        >>> if not storage_reader:
        >>>     return
        >>> dist_cp.load_state_dict(
        >>>     state_dict=state_dict,
        >>>     storage_reader=storage_reader,
        >>> )
        >>> model.load_state_dict(state_dict["model"])
        >>> optim_state = load_sharded_optimizer_state_dict(
        >>>     model_state_dict=state_dict["model"],
        >>>     optimizer_key="optim",
        >>>     storage_reader=storage_reader,
        >>> )
        >>> flattened_osd = FSDP.optim_state_dict_to_load(
        >>>     model, optimizer, optim_state["optim"]
        >>> )
        >>> optimizer.load_state_dict(flattened_osd)
    """

    def __init__(self, checkpoint_dir: str, storage=None, comm_backend=""):
        self.storage = PosixDiskStorage() if not storage else storage
        self._engine = FsdpCheckpointEngine(
            checkpoint_dir, self.storage, comm_backend
        )

    def save_checkpoint(
        self, step, state_dict, path, storage_type=StorageType.DISK
    ):
        paths = {CheckpointConstant.MODEL_STATES_NAME: path}
        if storage_type == StorageType.MEMORY:
            self._engine.save_to_memory(step, state_dict, paths)
        elif storage_type == StorageType.DISK:
            if not path:
                raise ValueError(
                    "path cannot be empty if storage type is disk!"
                )
            self._engine.save_to_storage(step, state_dict, path)
        else:
            raise ValueError(f"No support storage type {storage_type}")

    def load_checkpoint(self, resume_path=""):
        pass

    def get_storage_reader(self, resume_path=""):
        return self._engine.load(resume_path)
