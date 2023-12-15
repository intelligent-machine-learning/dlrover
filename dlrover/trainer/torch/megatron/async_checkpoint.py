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
from megatron import get_args

from dlrover.python.common.log import default_logger as logger

try:
    from megatron.checkpointing import (
        get_checkpoint_tracker_filename,
        load_checkpoint,
        save_checkpoint,
    )
except ImportError:
    logger.warning("Please check the magatron.checkpointing exists.")

from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    MegatronCheckpointEngine,
)


@singleton
class DlroverCheckpointSaver(object):
    def __init__(self, checkpoint_dir):
        self.state_dict = {}
        self.path = ""
        self.engine = MegatronCheckpointEngine(checkpoint_dir)
        self._tracer_file = get_checkpoint_tracker_filename(checkpoint_dir)
        self._latest_ckpt_iteration = 0

    def save(self, state_dict, path):
        self.state_dict = state_dict
        self.path = path

    def load(self, path):
        state_dict = self.engine.load(resume_path=path)
        return state_dict

    def clear_empty_checkpoint(self, iteration):
        ckpt_dir = os.path.join(self.engine.checkpoint_dir, iteration)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        with open(self._tracer_file, "w") as f:
            f.write(self._latest_ckpt_iteration)

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


def save_checkpoint_to_storage(
    iteration, model, optimizer, opt_param_scheduler
):
    """
    Asynchronously save the the checkpointing state dict into the storage.
    The method will not wait for saving the checkpointing to the storage.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    torch_save_func = torch.save
    torch.save = saver.save
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    saver.engine.save_to_storage(iteration, saver.state_dict, saver.path)
    torch.save = torch_save_func


def save_checkpoint_to_memory(
    iteration, model, optimizer, opt_param_scheduler
):
    """
    Synchronously save the the checkpointing state dict into the CPU memory.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    saver.update_latest_checkpoint_step()
    torch_save_func = torch.save
    torch.save = saver.save
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    saver.engine.save_to_memory(iteration, saver.state_dict, saver.path)
    torch.save = torch_save_func

    # Megatron save_checkpoint will create the directory with the iteration
    # and write the iteration into the tracerfile. But async saver only
    # save the state dict into the CPU memory not the storage. The saver
    # need to clear the empty checkpoint directory.
    saver.clear_empty_checkpoint()


def load_checkpoint_(
    model, optimizer, opt_param_scheduler, load_arg="load", strict=True
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    torch_load_func = torch.load
    torch.load = saver.load
    load_checkpoint(model, optimizer, opt_param_scheduler, load_arg, strict)
    torch.load = torch_load_func
