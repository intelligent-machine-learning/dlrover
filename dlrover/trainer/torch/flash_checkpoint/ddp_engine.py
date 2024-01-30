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

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    DdpCheckpointSaver,
)

from .engine import CheckpointEngine, timer


class DdpCheckpointEngine(CheckpointEngine):
    """
    Save the checkpoint state dict of DDP model into the memory or storage.

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

    def __init__(
        self,
        checkpoint_dir,
        storage,
        local_shard_num=1,
        global_shard_num=1,
        comm_backend="",
    ):
        if global_shard_num < local_shard_num:
            global_shard_num = local_shard_num
            logger.info(f"Set global_shard_num to {local_shard_num}.")
        self._local_shard_num = local_shard_num
        self._global_shard_num = global_shard_num
        super().__init__(checkpoint_dir, storage, comm_backend)

    def get_saving_ranks(self):
        """
        Only the local rank 0 in each node saves the state dict into the
        memory. They need to synchronize the saving status.
        """
        group_size = env_utils.get_group_world_size()
        local_world_size = env_utils.get_local_world_size()
        save_ranks = []
        for i in range(group_size):
            for j in range(self._local_shard_num):
                saver_rank = i * local_world_size + j
                save_ranks.append(saver_rank)
        logger.info(f"The ranks to save checkpoint are {save_ranks}.")
        return save_ranks

    def get_local_shard_num(self):
        return self._local_shard_num

    def get_global_shard_num(self):
        return self._global_shard_num

    def get_saver_class(self):
        return DdpCheckpointSaver

    @timer
    def save_to_memory(self, step, state_dict, paths: Dict[str, str]):
        """
        Synchronously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a category in
                ["model_states", "optim_states"] of the state dict and
                the value is the path of storage to save.
        """
        conf = CheckpointConfig(step=step, paths=paths)
        return self.save_state_dict_to_memory(state_dict, conf)

    @timer
    def save_to_storage(self, step, state_dict, paths):
        """
        Asynchronously saves the state dict into the storage. It synchronously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a category in
                ["model_states", "optim_states"] of the state dict and
                the value is the path of storage to save.
        """
        succeed = True
        if step > self._cached_step:
            succeed = self.save_to_memory(step, state_dict, paths)
        # Only rank 0 persist the checkpoint to the storage.
        if dist.is_initialized():
            dist.barrier()
        if succeed and self._rank == 0:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
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
            paths = list(state_dict.keys())
            if len(paths) > 1:
                raise ValueError(
                    "The checkpoint shared memory must has only the"
                    f"state dict of one path. Now, paths are {paths}"
                )
            path = paths[0]
            return state_dict[path]
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
            state_dict = self.storage.read_state_dict(
                resume_path,
                read_func=lambda path: torch.load(path, map_location="cpu"),
            )
            return state_dict
        else:
            tracker_filename = os.path.join(
                self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
            )
            content: str = self.storage.read(tracker_filename)
            if not content:
                return state_dict
            iteration = int(content.strip())
            if self._global_shard_num == 1:
                #  Load the checkpoint saved by rank 0 if no sharding.
                name = f"{iteration}/rank_{self._rank}.pt"
            else:
                name = f"{iteration}/rank_0.pt"
            path = os.path.join(self.checkpoint_dir, name)
            logger.info(f"Load the state dict from {path}")
            state_dict = self.storage.read_state_dict(
                path,
                read_func=lambda path: torch.load(path, map_location="cpu"),
            )
            return state_dict
