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

import os
import shutil

import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger

try:
    from megatron import get_args
    from megatron.checkpointing import get_checkpoint_tracker_filename
    from megatron.checkpointing import load_checkpoint as megatron_load
    from megatron.checkpointing import save_checkpoint as megatron_save
except ImportError:
    logger.warning("Please check the magatron.checkpointing exists.")

from dlrover.python.common.singleton import singleton

from .checkpointer import StorageType
from .megatron_engine import MegatronCheckpointEngine


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


@singleton
class MegatronCheckpointManager(object):
    def __init__(self, checkpoint_dir):
        self.state_dict = {}
        self.path = ""
        self.engine = MegatronCheckpointEngine(checkpoint_dir)
        self._tracer_file = get_checkpoint_tracker_filename(checkpoint_dir)
        self._latest_ckpt_iteration = 0

    def save(self, state_dict, path):
        self.state_dict = state_dict
        self.path = path

    def load(self, path, **kwargs):
        state_dict = self.engine.load(resume_path=path)
        return state_dict

    def clear_empty_checkpoint(self, iteration):
        ckpt_dir = os.path.join(
            self.engine.checkpoint_dir, "iter_{:07d}".format(iteration)
        )
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        if self._latest_ckpt_iteration > 0:
            with open(self._tracer_file, "w") as f:
                f.write(str(self._latest_ckpt_iteration))
        elif self._latest_ckpt_iteration == 0 and os.path.exists(
            self._tracer_file
        ):
            os.remove(self._tracer_file)

    def update_latest_checkpoint_step(self):
        if not os.path.exists(self._tracer_file):
            return
        iteration = 0
        with open(self._tracer_file, "r") as f:
            try:
                content = f.read().strip()
                iteration = int(content)
            except Exception as e:
                logger.warning(e)
        if iteration > 0:
            self._latest_ckpt_iteration = iteration


def save_checkpoint(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    storage_type=StorageType.DISK,
):
    """
    Synchronously save the the checkpointing state dict into the CPU memory.

    Args:
        same as the `megatron.checkpointing.load_checkpoint`
    """
    if storage_type == StorageType.MEMORY:
        args = get_args()
        saver = MegatronCheckpointManager(args.save)
        saver.update_latest_checkpoint_step()
        torch_save_func = torch.save
        torch.save = saver.save
        megatron_save(iteration, model, optimizer, opt_param_scheduler)
        saver.engine.save_to_memory(iteration, saver.state_dict, saver.path)
        torch.save = torch_save_func

        # Megatron save_checkpoint will create the directory with the iteration
        # and write the iteration into the tracerfile. But async saver only
        # save the state dict into the CPU memory not the storage. The saver
        # need to clear the empty checkpoint directory.
        if _get_rank() == 0:
            saver.clear_empty_checkpoint(iteration)
    elif storage_type == StorageType.DISK:
        args = get_args()
        saver = MegatronCheckpointManager(args.save)
        torch_save_func = torch.save
        torch.save = saver.save
        megatron_save(iteration, model, optimizer, opt_param_scheduler)
        saver.engine.save_to_storage(iteration, saver.state_dict, saver.path)
        torch.save = torch_save_func
    else:
        raise ValueError(f"No support storage type {storage_type}")


def load_checkpoint(
    model, optimizer, opt_param_scheduler, load_arg="load", strict=True
):
    """Load the checkpointing state dict. The method firstly
    load the state dict from the CPU memory and then from the storage.
    Args:
        same as the `megatron.checkpointing.load_checkpoint`
    """
    args = get_args()
    saver = MegatronCheckpointManager(args.save)
    torch_load_func = torch.load
    torch.load = saver.load
    iteration = megatron_load(
        model, optimizer, opt_param_scheduler, load_arg, strict
    )
    torch.load = torch_load_func
    return iteration
