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
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.storage import get_checkpoint_storage
from dlrover.trainer.torch.flash_checkpoint.full_ckpt_engine import (
    FullCheckpointEngine,
)

from .checkpointer import Checkpointer, StorageType
from .fsdp_engine import FsdpCheckpointEngine


class FsdpShardCheckpointer(Checkpointer):
    """
    Flash checkpointer saves and loads a FSDP module.

    Args:
        checkpoint_dir: the directory to save the checkpoint.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
        deletion_strategy: A `CheckpointDeletionStrategy` instance. The default
            value is None and all checkpoint files will be retained. Now, the
            strategy can be `KeepLatestStepStrategy`
            or `KeepStepIntervalStrategy`. Users also can define a strategy
            to manage the checkpoint files.

    Examples::
        >>> checkpointer = FsdpShardCheckpointer(checkpoint_dir)
        >>> # Save checkpoint
        >>> extra_sd = {"step": step, "epoch": epoch}
        >>> if step % save_storage_interval == 0:
        >>>     checkpointer.save_checkpoint(
        >>>         step, state_dict, ckpt_dir
        >>>     )
        >>> # Load checkpoint
        >>> checkpointer.load_checkpoint(model, optimzier)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        comm_backend="",
        deletion_strategy=None,
        save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.storage = get_checkpoint_storage(deletion_strategy)
        self._engine = FsdpCheckpointEngine(
            checkpoint_dir, self.storage, comm_backend, save_timeout
        )

    def save_checkpoint(
        self,
        step,
        model,
        optimizer,
        extra_sd={},
        path="",
        storage_type=StorageType.DISK,
    ):
        """
        Save a fsdp model and optimizer.

        Args:
            step(int): the iteration step.
            model: A FSDP module.
            optimizer: An optimizer to train a FSDP model.
            extra_sd(dict): A dict to store customized arguments
                in the checkpoint.
            path(str): A path to store the checkpoint.
            storage_type: Save the checkpoint into the memory
                if `StorageType.MEMORY` and into the dist
                if `StorageType.DISK`.
        """
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optimizer),
            }
            state_dict.update(extra_sd)
            if not path:
                path = os.path.join(self.checkpoint_dir, str(step))
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

    def load_checkpoint(self, model, optimizer, resume_path=""):
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "step": 0,
                # cannot load the optimizer state_dict together
                # with the model state_dict
            }
            storage_reader = self._engine.load(resume_path)
            if not storage_reader:
                return {}
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=storage_reader,
            )
            model_sd = state_dict.pop("model", None)
            if model_sd:
                model.load_state_dict(model_sd)

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=model_sd,
                optimizer_key="optim",
                storage_reader=storage_reader,
            )

            optim_sd = optim_state.pop("optim", None)
            if optim_sd:
                flattened_osd = FSDP.optim_state_dict_to_load(
                    model, optimizer, optim_sd
                )
                optimizer.load_state_dict(flattened_osd)
            return state_dict


class FsdpFullCheckpointer(Checkpointer):
    """
    Flash checkpointer to save and load a FSDP full model.

    Args:
        checkpoint_dir: the directory to save the checkpoint.
        comm_backend (str): the communcation backend to create a process group,
            The default is the backend of general main process group.
        deletion_strategy: A `CheckpointDeletionStrategy` instance. The default
            value is None and all checkpoint files will be retained. Now, the
            strategy can be `KeepLatestStepStrategy`
            or `KeepStepIntervalStrategy`. Users also can define a strategy
            to manage the checkpoint files.

    Examples::
        >>> checkpointer = FsdpFullCheckpointer(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>> )
        >>> for step, data in enumerate(dataloader):
        >>>     ...
        >>>     extra_sd = {"epoch": 10}
        >>>     path = f"/tmp/checkpoint-{step}.pt"
        >>>     if step % 100 == 0:
        >>>         checkpointer.save_checkpoint(
        >>>             step, model, optimizer, extra_sd, path
        >>>         )
        >>> sate_dict = checkpointer.load_checkpoint(model, optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        comm_backend="",
        deletion_strategy=None,
        save_timeout: int = CheckpointConstant.SAVE_TIMEOUT,
    ):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
        else:
            self._rank = 0
        self.storage = get_checkpoint_storage(deletion_strategy)
        self._engine = FullCheckpointEngine(
            checkpoint_dir=checkpoint_dir,
            storage=self.storage,
            local_shard_num=1,
            global_shard_num=1,
            comm_backend=comm_backend,
            save_timeout=save_timeout,
        )

    def save_checkpoint(
        self,
        step,
        model,
        optimizer,
        extra_sd={},
        path="",
        storage_type=StorageType.DISK,
    ):
        """
        Save a fsdp model and optimizer.

        Args:
            step(int): the iteration step.
            model: A FSDP module.
            optimizer: An optimizer to train a FSDP model.
            extra_sd(dict): A dict to store customized arguments
                in the checkpoint.
            path(str): A path to store the checkpoint.
            storage_type: Save the checkpoint into the memory
                if `StorageType.MEMORY` and into the dist
                if `StorageType.DISK`.
        """
        if path == "":
            ckpt_name = f"{step}/rank_{self._rank}.pt"
            path = os.path.join(self.checkpoint_dir, ckpt_name)

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        ):
            msd = model.state_dict()
            osd = FSDP.optim_state_dict(model, optimizer)
        state_dict = {"model": msd, "optimizer": osd}
        state_dict.update(extra_sd)

        state_dict = {CheckpointConstant.MODEL_STATES_NAME: state_dict}
        paths = {CheckpointConstant.MODEL_STATES_NAME: path}
        if storage_type == StorageType.MEMORY:
            self._engine.save_to_memory(step, state_dict, paths)
        elif storage_type == StorageType.DISK:
            if not path:
                raise ValueError(
                    "path cannot be empty if storage type is disk!"
                )
            if self._rank == 0:
                self.storage.safe_rmtree(os.path.dirname(path))
            self._engine.save_to_storage(step, state_dict, paths)
        else:
            raise ValueError(f"No support storage type {storage_type}")

    def load_checkpoint(self, model, optimizer, resume_path=""):
        """
        Load a fsdp model and optimizer.

        Args:
            model: A FSDP module.
            optimizer: An optimizer to train a FSDP model.
            resume_path (str): the path to restore the checkpoint.
        """
        state_dict = self._engine.load(resume_path)
        if not state_dict:
            return {}
        model_state_dict = state_dict.pop("model", {})
        optim_state_dict = state_dict.pop("optimizer", {})

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        ):
            optim_state_dict = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=optim_state_dict,
            )
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        return state_dict
