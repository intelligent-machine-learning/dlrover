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

"""Input/output checkpointing."""

import inspect
import os

import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger

try:
    from megatron import get_args
    from megatron.checkpointing import load_checkpoint as megatron_load
    from megatron.checkpointing import save_checkpoint as megatron_save
except ImportError:
    logger.warning("Please check the magatron.checkpointing exists.")

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.singleton import singleton
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    MegatronCheckpointSaver,
)

from .checkpointer import StorageType
from .megatron_engine import MegatronCheckpointEngine

_MODEL_SD_NAME = "model_optim_rng.pt"
_DIST_OPTIM_SD_NAME = "distrib_optim.pt"

torch_native_save = torch.save
torch_native_load = torch.load


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


@singleton
class MegatronCheckpointManager(object):
    def __init__(self, checkpoint_dir, storage=None, comm_backend=""):
        self.state_dict = {}
        self.paths = {}
        self.checkpoint_dir = checkpoint_dir
        self.storage = PosixDiskStorage() if not storage else storage
        self.engine = MegatronCheckpointEngine(
            checkpoint_dir=checkpoint_dir,
            storage=self.storage,
            comm_backend=comm_backend,
        )

    def save(self, state_dict, path: str):
        if not isinstance(path, str):
            torch_native_save(state_dict, path)
            return
        if path.endswith(_MODEL_SD_NAME):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DIST_OPTIM_SD_NAME):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        else:
            raise ValueError(
                "MegatronCheckpointer only support the path whose suffix is "
                f"{_MODEL_SD_NAME} or {_DIST_OPTIM_SD_NAME}."
            )
        self.state_dict[sd_name] = state_dict
        self.paths[sd_name] = path

    def load(self, path: str, **kwargs):
        def load_func(path):
            return torch_native_load(path, map_location="cpu")

        if not isinstance(path, str):
            return torch_native_load(path)

        state_dict = self.engine.load(resume_path=path)

        sd_name = ""
        if path.endswith(_MODEL_SD_NAME):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DIST_OPTIM_SD_NAME):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        if sd_name in state_dict:
            return state_dict[sd_name]
        else:
            return self.storage.read_state_dict(path, load_func)

    def update_tracer_file(self, iteration: int):
        """
        Update the tracer file with the latest step saved in the storage.
        The `save_checkpoint` of Megatron will modify the tracer file even
        if the storage_type is StorageType.Memory. We need to restore
        the tracer file  after saving the checkpoint into the memory.

        Args:
            iteration (int): the iteration step to save.
        """
        ckpt_dir = os.path.join(
            self.checkpoint_dir, "iter_{:07d}".format(iteration)
        )
        self.storage.safe_rmtree(ckpt_dir)

        megatron_tracer_file = os.path.join(
            self.checkpoint_dir, MegatronCheckpointSaver.TRACER_FILE
        )

        dlrover_tracer_file = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        content = self.storage.read(dlrover_tracer_file)
        if content:
            self.storage.write(content, megatron_tracer_file)
        else:
            self.storage.safe_remove(megatron_tracer_file)


def save_checkpoint(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far=0,
    storage_type=StorageType.DISK,
    storage=None,
    comm_backend="",
):
    """
    Synchronously save the the checkpointing state dict into the CPU memory.

    Args:
        same as the `megatron.checkpointing.load_checkpoint`
        storage: A CheckpointStorage instance. The checkpointer will
            use a PosixStorage instance if the storage is not defined.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
    """
    args = get_args()
    saver = MegatronCheckpointManager(
        args.save, storage=storage, comm_backend=comm_backend
    )
    sig = inspect.signature(megatron_save)
    if storage_type == StorageType.MEMORY:
        torch.save = saver.save
        try:
            if "num_floating_point_operations_so_far" in sig.parameters:
                megatron_save(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                )
            else:
                megatron_save(iteration, model, optimizer, opt_param_scheduler)
        finally:
            torch.save = torch_native_save
            # Megatron save_checkpoint will create the directory with the
            # iteration and write the iteration into the tracerfile. But async
            # saver only save the state dict into the CPU memory not
            # the storage. The saver need to clear the empty checkpoint
            # directory.
            if _get_rank() == 0:
                saver.update_tracer_file(iteration)
        saver.engine.save_to_memory(iteration, saver.state_dict, saver.paths)
    elif storage_type == StorageType.DISK:
        try:
            torch.save = saver.save
            if "num_floating_point_operations_so_far" in sig.parameters:
                megatron_save(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                )
            else:
                megatron_save(iteration, model, optimizer, opt_param_scheduler)
        finally:
            torch.save = torch_native_save
            if _get_rank() == 0:
                saver.update_tracer_file(iteration)
        saver.engine.save_to_storage(iteration, saver.state_dict, saver.paths)
    else:
        raise ValueError(f"No support storage type {storage_type}")

    saver.state_dict.clear()


def load_checkpoint(
    model,
    optimizer,
    opt_param_scheduler,
    load_arg="load",
    strict=True,
    comm_backend="",
):
    """Load the checkpointing state dict. The method firstly
    load the state dict from the CPU memory and then from the storage.
    Args:
        same as the `megatron.checkpointing.load_checkpoint`
    """
    args = get_args()
    saver = MegatronCheckpointManager(args.save, comm_backend=comm_backend)
    torch.load = saver.load
    iteration = megatron_load(
        model, optimizer, opt_param_scheduler, load_arg, strict
    )
    torch.load = torch_native_load
    return iteration
