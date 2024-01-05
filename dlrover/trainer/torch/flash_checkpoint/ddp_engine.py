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
from datetime import timedelta

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    CheckpointEvent,
    CheckpointEventType,
    DdpCheckpointSaver,
)

from .engine import CheckpointEngine, timer


class DdpCheckpointEngine(CheckpointEngine):
    """
    Save the checkpoint state dict of DDP model into the memory or storage.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.

    Examples::
        >>> engine = DdpCheckpointEngine(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>> )
        >>> for step, data in enumerate(dataloader):
        >>>     ...
        >>>     state_dict = model.state_dict()
        >>>     path = f"/tmp/checkpoint-{step}.pt"
        >>>     if step % 5 == 0:
        >>>         engine.save_to_memory(step, state_dict, path)
        >>>     elif step % 100 == 0:
        >>>         engine.save_to_storage(step, state_dict, path)
        >>> sate_dict = engine.load()
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir)
        if dist.is_initialized():
            saver_ranks = self._get_saver_ranks()
            self._saver_group = dist.new_group(
                ranks=saver_ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )

    def _get_saver_ranks(self):
        """
        Only the local rank 0 in each node saves the state dict into the
        memory. They need to synchronize the saving status.
        """
        group_size = env_utils.get_group_world_size()
        local_world_size = env_utils.get_local_world_size()
        save_ranks = []
        for i in range(group_size):
            saver_rank = i * local_world_size
            save_ranks.append(saver_rank)
        logger.info(f"The ranks to save checkpoint are {save_ranks}.")
        return save_ranks

    def get_local_shard_num(self):
        return 1

    def get_global_shard_num(self):
        return 1

    def get_saver_class(self):
        return DdpCheckpointSaver

    @timer
    def save_to_storage(self, step, state_dict, path=""):
        """
        Asynchronously saves the state dict into the storage. It synchronously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): the storage path to save the state dict.
                Note, the ckpt_name is used to save the state dict to storage
                only if the training process fails.
        """
        if self._local_rank != 0:
            return
        if not path:
            name = f"{CheckpointConstant.CKPT_NAME_PREFIX}{step}.pt"
            path = os.path.join(self.checkpoint_dir, name)
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, path)
        event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)

        # Only rank 0 persist the checkpoint to the storage.
        self._event_queue.put(event)

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self.get_state_dict_from_memory()
        if state_dict:
            logger.info("Load the state dict from the CPU memory buffer.")
            state_dict.pop(DLROVER_CKPT_CONFIG_KEY, None)
            return state_dict
        state_dict = self._load_from_storage(resume_path)
        return state_dict

    def _load_from_storage(self, resume_path=""):
        """
        Load the state dict from the CPU memory if the state dict is complete
        in CPU memory. Otherwise, the function will load the state dict from
        the storage.

        Args:
            resume_path (str, optional): , If the resume_path is an empty
                string, the function will load the latest checkpoint file in
                the checkpoint directory.

        Returns:
            A dict:
                a dictionary containing a whole state of the modules in the
                checkpointing file.
        """
        state_dict = {}
        if resume_path:
            state_dict = torch.load(resume_path, map_location="cpu")
            return state_dict
        else:
            tracker_filename = os.path.join(
                self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
            )
            if not os.path.exists(tracker_filename):
                return state_dict
            with open(tracker_filename, "r") as f:
                metastring = f.read().strip()
            iteration = int(metastring)
            name = f"{CheckpointConstant.CKPT_NAME_PREFIX}{iteration}.pt"
            path = os.path.join(self.checkpoint_dir, name)
            if not os.path.exists(path):
                logger.warning(f"Checkpoint path {path} is not exist.")
                return state_dict
            logger.info(f"Load the state dict from {path}")
            state_dict = torch.load(path, map_location="cpu")
            return state_dict
