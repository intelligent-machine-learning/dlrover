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
from typing import Dict

import torch
import torch.distributed as dist
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
    CheckpointEngine,
)
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.config import ZeroStageEnum

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.storage import (
    CheckpointStorage,
    get_checkpoint_storage,
)
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DeepSpeedCheckpointSaver,
)

from .checkpointer import Checkpointer, StorageType
from .deepspeed_engine import DeepSpeedCheckpointEngine

_DS_MODEL_SD_FILE_SUFFIX = "model_states.pt"
_DS_OPTIM_SD_FILE_SUFFIX = "optim_states.pt"

torch_native_save = torch.save
torch_native_load = torch.load


class AsyncCheckpointAgent(CheckpointEngine):
    """
    The checkpoint agent to save/load checkpoint of DeepSpeed.

    Attributes:
        model_sd: the state dict of a PyTorch model.
        model_path (str): the storage path to save the model state dict.
        optim_sd: the state dict of a DeepSpeed optimizer.
        optim_path (str): the storage path to save the optim state dict.
    """

    def __init__(self, storage: CheckpointStorage):
        self.state_dict: Dict[str, object] = {}
        self.paths: Dict[str, str] = {}
        self.storage = storage

    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def save(self, state_dict, path: str):
        if not isinstance(path, str):
            torch_native_save(state_dict, path)
            return
        if path.endswith(_DS_MODEL_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DS_OPTIM_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        else:
            sd_name = path.split("/")[-1]
        if sd_name:
            self.state_dict[sd_name] = state_dict
            self.paths[sd_name] = path

    def load(self, path: str, map_location=None):
        def load_func(path):
            return torch_native_load(path, map_location=map_location)

        sd_name = ""
        if path.endswith(_DS_MODEL_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DS_OPTIM_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        if sd_name in self.state_dict:
            return self.state_dict[sd_name]
        else:
            return self.storage.read_state_dict(path, load_func)

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass


class DeepSpeedCheckpointer(Checkpointer):
    """
    Flash checkpointer saves and loads a DeepSpeedEngine module.

    Args:
        checkpoint_dir: the directory to save the checkpoint.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
        deletion_strategy: A `CheckpointDeletionStrategy` instance. The default
            value is None and all checkpoint files will be retained. Now, the
            strategy can be `KeepLatestStepStrategy`
            or `KeepStepIntervalStrategy`. Users also can define a strategy
            to manage the checkpoint files.
        save_timeout (int): the seconds for node rank 0 to wait all
            ranks save checkpoints. The node rank 0 will skip the checkpoint
            if some ranks do not finish saving checkpoint in the save_timeout
            after the node rank 0 finishes saving checkpoint.

    Examples::
        >>> engine = deepspeed.initialize(...)
        >>> checkpointer = DeepSpeedCheckpointer(engine, save_dir)
        >>> if step % 10 == 0:
        >>>     checkpointer.save_checkpoint(
        >>>         save_dir, tag, storage_type=StorageType.MEMORY
        >>>     )
        >>> if step % 100 == 0:
        >>>     checkpointer.save_checkpoint(
        >>>         save_dir, tag, storage_type=StorageType.DISK
        >>>     )
    """

    def __init__(
        self,
        engine: DeepSpeedEngine,
        checkpoint_dir,
        comm_backend="",
        deletion_strategy=None,
        save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    ):
        self.engine = engine
        self.checkpoint_dir = checkpoint_dir
        global_shard_num = 1
        if self.engine.zero_optimization():
            global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )
        zero_stage = self.engine.zero_optimization_stage()
        self.storage = get_checkpoint_storage(deletion_strategy)
        self._async_save_engine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            global_shard_num=global_shard_num,
            zero_stage=zero_stage,
            comm_backend=comm_backend,
            save_timeout=save_timeout,
        )
        self._ckpt_agent = AsyncCheckpointAgent(
            self._async_save_engine.storage
        )
        self._local_rank = env_utils.get_local_rank()
        self._ds_tracer_file = os.path.join(
            self.checkpoint_dir, DeepSpeedCheckpointSaver.TRACER_FILE
        )
        self._dlrover_tracer_file = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        if zero_stage < ZeroStageEnum.weights and self._local_rank == 0:
            self.engine.save_non_zero_checkpoint = True

    def save_checkpoint(
        self,
        save_dir,
        tag=None,
        client_state={},
        save_latest=True,
        storage_type=StorageType.DISK,
    ):
        if storage_type == StorageType.MEMORY:
            self._save_checkpoint_to_memory(
                save_dir, tag, client_state, save_latest
            )
        elif storage_type == StorageType.DISK:
            self._save_checkpoint_to_storage(
                save_dir, tag, client_state, save_latest
            )
        else:
            raise ValueError(f"No support storage type {storage_type}")

    def _save_checkpoint_to_memory(
        self, save_dir, tag=None, client_state={}, save_latest=True
    ):
        torch.save = self._ckpt_agent.save
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        torch.save = torch_native_save
        self._async_save_engine.save_to_memory(
            tag,
            self._ckpt_agent.state_dict,
            self._ckpt_agent.paths,
        )
        self._update_tracer_file(tag)

    def _update_tracer_file(self, tag):
        """
        The method save_to_memory does not save the state dict into
        the storage. We need to restore the tracer file modified
        by DeepSpeedEngine.save_checkpoint after calling save to
        the memory.
        """
        if self.engine.global_rank != 0:
            return
        ckpt_dir = os.path.join(self.checkpoint_dir, str(tag))
        self.storage.safe_rmtree(ckpt_dir)
        content = self.storage.read(self._dlrover_tracer_file)
        if content:
            self.storage.write(content, self._ds_tracer_file)
        else:
            self.storage.safe_remove(self._ds_tracer_file)

    def _save_checkpoint_to_storage(
        self, save_dir, tag=None, client_state={}, save_latest=True
    ):
        torch.save = self._ckpt_agent.save
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        torch.save = torch_native_save
        self._async_save_engine.save_to_storage(
            tag,
            self._ckpt_agent.state_dict,
            self._ckpt_agent.paths,
        )

    def load_checkpoint(
        self,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        custom_load_fn=None,
    ):
        """
        Load a checkpointing state dict.

        Args:
            the same as the DeepSpeedEngine.load_checkpoint.
        """
        self._ckpt_agent.state_dict = self._async_save_engine.load()
        torch.load = self._ckpt_agent.load
        load_path, client_states = self.engine.load_checkpoint(
            load_dir=load_dir,
            tag=tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
            custom_load_fn=custom_load_fn,
        )
        torch.load = torch_native_load
        return load_path, client_states
