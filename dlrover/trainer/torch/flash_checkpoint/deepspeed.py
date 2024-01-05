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
import shutil

import torch
import torch.distributed as dist
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
    CheckpointEngine,
)
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.config import ZeroStageEnum
from torch.serialization import load

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DeepSpeedCheckpointSaver,
)

from .checkpointer import Checkpointer, StorageType
from .deepspeed_engine import DeepSpeedCheckpointEngine


class AsyncSaveEngine(CheckpointEngine):
    def __init__(self):
        self.model_sd = None
        self.model_path = ""
        self.optimizer_sd = None
        self.optimizer_path = ""

    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def save(self, state_dict, path: str):
        if CheckpointConstant.MODEL_STATES_NAME in path:
            self.model_sd = state_dict
            self.model_path = path
        elif CheckpointConstant.OPTIM_STATES_NAME in path:
            self.optimizer_sd = state_dict
            self.optimizer_path = path

    def load(self, path: str, map_location=None):
        if CheckpointConstant.MODEL_STATES_NAME in path:
            if self.model_sd:
                return self.model_sd
            else:
                return load(path, map_location=map_location)
        elif CheckpointConstant.OPTIM_STATES_NAME in path:
            if self.optimizer_sd:
                return self.optimizer_sd
            else:
                return load(path, map_location=map_location)

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass


class DeepSpeedCheckpointer(Checkpointer):
    """
    Flash checkpointer saves and loads a DeepSpeedEngine module.

    Args:
        checkpoint_dir: the directory to save the checkpoint.

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

    def __init__(self, engine: DeepSpeedEngine, checkpoint_dir):
        self.engine = engine
        self.checkpoint_dir = checkpoint_dir
        self._ckpt_engine = AsyncSaveEngine()
        self.engine.checkpoint_engine = self._ckpt_engine
        global_shard_num = 1
        if self.engine.zero_optimization():
            global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )
        zero_stage = self.engine.zero_optimization_stage()
        self._async_save_engine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            global_shard_num=global_shard_num,
            zero_stage=zero_stage,
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
        torch_save_func = torch.save
        torch.save = self._ckpt_engine.save
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        torch.save = torch_save_func
        state_dict = self._merge_model_and_optmizer_state_dict()
        self._async_save_engine.save_to_memory(
            tag,
            state_dict,
            model_path=self._ckpt_engine.model_path,
            optimizer_path=self._ckpt_engine.optimizer_path,
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
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        if os.path.exists(self._dlrover_tracer_file):
            with open(self._dlrover_tracer_file, "r") as f:
                step = f.read()
            with open(self._ds_tracer_file, "w") as f:
                f.write(step)
        elif os.path.exists(self._ds_tracer_file):
            os.remove(self._ds_tracer_file)

    def _save_checkpoint_to_storage(
        self, save_dir, tag=None, client_state={}, save_latest=True
    ):
        torch_save_func = torch.save
        torch.save = self._ckpt_engine.save
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        torch.save = torch_save_func
        state_dict = self._merge_model_and_optmizer_state_dict()
        self._async_save_engine.save_to_storage(
            tag,
            state_dict,
            model_path=self._ckpt_engine.model_path,
            optimizer_path=self._ckpt_engine.optimizer_path,
        )

    def _merge_model_and_optmizer_state_dict(self):
        merged_state_dict = {}
        if self._ckpt_engine.model_sd:
            merged_state_dict[
                CheckpointConstant.MODEL_STATES_NAME
            ] = self._ckpt_engine.model_sd
        if self._ckpt_engine.optimizer_sd:
            merged_state_dict[
                CheckpointConstant.OPTIM_STATES_NAME
            ] = self._ckpt_engine.optimizer_sd
        return merged_state_dict

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
        state_dict = self._async_save_engine.load()
        self._ckpt_engine.model_sd = state_dict.get(
            CheckpointConstant.MODEL_STATES_NAME, {}
        )
        self._ckpt_engine.optimizer_sd = state_dict.get(
            CheckpointConstant.OPTIM_STATES_NAME, {}
        )
        torch_load_func = torch.load
        torch.load = self._ckpt_engine.load
        load_path, client_states = self.engine.load_checkpoint(
            load_dir=load_dir,
            tag=tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
            custom_load_fn=custom_load_fn,
        )
        torch.load = torch_load_func
        return load_path, client_states

    def _check_latest(self):
        """
        DeepSpeed engine may overwrite the latest file after calling
        `save_checkpoint` which only saves the checkpointing state dict
        into the momery. So, the latest step in "dlrover_latset" is
        always correct after the dlrover agent saves the checkpointint
        state dict to the storage.
        """
        dlrover_file = os.path.join(self.checkpoint_dir, "dlrover_latest")
        if not os.path.exists(dlrover_file):
            return
        with open(dlrover_file, "r") as f:
            step = f.read()

        ds_file = os.path.join(self.checkpoint_dir, "latest")
        with open(ds_file, "w") as f:
            f.write(step)
