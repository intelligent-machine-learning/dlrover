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

import torch.distributed as dist

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.storage import PosixDiskStorage

from .checkpointer import Checkpointer, StorageType
from .ddp_engine import DdpCheckpointEngine


class DdpCheckpointer(Checkpointer):
    """
    Flash checkpointer to save and load a DDP model.

    Args:
        checkpoint_dir: the directory to save the checkpoint.
        storage: A CheckpointStorage instance. The checkpointer will
            use a PosixStorage instance if the storage is not defined.
        local_shard_num (int): the number of shards on a node,
            The default is 1. If the model is partitioned on all ranks,
            you should set the local_shard_num as the number of ranks
            on a node.
        global_shard_num (int): the number of shards across all ranks.
            The default is 1.If the model is partitioned on all ranks,
            you should set the local_shard_num as the number of all ranks.
        comm_backend (str): the communcation backend to create a process group,
            The default is the backend of general main process group.

    Examples::
        >>> checkpointer = DdpCheckpointer(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>> )
        >>> for step, data in enumerate(dataloader):
        >>>     ...
        >>>     state_dict = model.state_dict()
        >>>     path = f"/tmp/checkpoint-{step}.pt"
        >>>     if step % 5 == 0:
        >>>         checkpointer.save_checkpoint(
        >>>             step, state_dict, path, storage_type=StorageType.Memory
        >>>         )
        >>>     elif step % 100 == 0:
        >>>         checkpointer.save_checkpoint(
        >>>             step, state_dict, path, storage_type=StorageType.DISK
        >>>         )
        >>> sate_dict = engine.load_checkpoint()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        storage=None,
        local_shard_num=1,
        global_shard_num=1,
        comm_backend="",
    ):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
        else:
            self._rank = 0
        self.storage = PosixDiskStorage() if not storage else storage
        self._engine = DdpCheckpointEngine(
            checkpoint_dir=checkpoint_dir,
            storage=self.storage,
            local_shard_num=local_shard_num,
            global_shard_num=global_shard_num,
            comm_backend=comm_backend,
        )

    def save_checkpoint(
        self, step, state_dict, path="", storage_type=StorageType.DISK
    ):
        if path == "":
            ckpt_name = f"{step}/rank_{self._rank}.pt"
            path = os.path.join(self.checkpoint_dir, ckpt_name)
        state_dict = {CheckpointConstant.MODEL_STATES_NAME: state_dict}
        paths = {CheckpointConstant.MODEL_STATES_NAME: path}
        if storage_type == StorageType.MEMORY:
            self._engine.save_to_memory(step, state_dict, paths)
        elif storage_type == StorageType.DISK:
            if not path:
                raise ValueError(
                    "path cannot be empty if storage type is disk!"
                )
            self._engine.save_to_storage(step, state_dict, paths)
        else:
            raise ValueError(f"No support storage type {storage_type}")

    def load_checkpoint(self, resume_path=""):
        return self._engine.load(resume_path)
