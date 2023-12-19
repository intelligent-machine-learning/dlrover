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

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    _DLROVER_CKPT_KEY,
    _SAVE_EVENT_NAME,
    CheckpointEngine,
    FsdpCheckpointSaver,
    SaveEvent,
    SingleFileCheckpointConfig,
    check_all_rank_ready,
    timer,
)

from .file_reader import FileReader
from .shared_memory import SharedMemoryReader, SharedMemoryWriter


class FsdpCheckpointManager(CheckpointEngine):
    def __init__(self, checkpoint_dir: str):
        super().__init__(checkpoint_dir)
        self._shm_writer = SharedMemoryWriter(shm_handler=self._shm_handler)
        self._shm_reader = SharedMemoryReader(self._shm_handler)

    @timer
    def save_to_memory(self, step, state_dict, path=""):
        """
        Synchronously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): the storage path to save the state dict.
                Note, the path is used to save the state dict to storage
                only if the training process fails.
        """
        if self._local_rank != self.local_shard_id:
            return

        conf = SingleFileCheckpointConfig(
            step=step,
            path=path,
        )

        acquired = self._shm_lock.acquire(blocking=False)
        all_rank_ready = check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_lock.release()
            return

        conf.writing_shm = True
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=self._shm_writer,
        )
        conf.path = os.path.join(path, self._shm_writer.file_name)
        meta_dict = {_DLROVER_CKPT_KEY: conf}
        meta_dict.update(self._shm_writer.metadata)
        self._shm_handler.metadata.set(meta_dict)
        conf.writing_shm = False
        if acquired:
            self._shm_lock.release()
        self._cached_step = conf.step

    def save_to_storage(self, step, state_dict, path):
        """
        Save the state_dict into the path of storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
        """
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, path)

        # Only local rank 0 on each node notifies the event to save.
        if self._local_rank != 0:
            return
        if path:
            event = SaveEvent(name=_SAVE_EVENT_NAME, step=step)
            self._event_queue.put(event)

    def get_saver_class(self):
        """
        Get a CheckpointSaver class.
        """
        return FsdpCheckpointSaver

    def get_local_shard_num(self):
        """Get the number of model shards on the node."""
        return env_utils.get_local_world_size()

    def get_global_shard_num(self):
        """Get the number of model shards on all nodes."""
        return dist.get_world_size()

    def load(self, resume_path="", model_state_dict={}):
        """
        Load the checkpointing state dict from the resume path.

        Returns:
            A dict.
        """
        pass

        if not self._shm_handler.empty():
            state_dict = {
                "model": model_state_dict,
                # cannot load the optimizer state_dict together with the
                # model state_dict
            }

            dist_cp.load_state_dict(
                state_dict=state_dict, storage_reader=self._shm_reader
            )

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=self._shm_reader,
            )
            return state_dict["model"], optim_state["optim"]
        else:
            state_dict = {
                "model": model_state_dict,
                # cannot load the optimizer state_dict together
                # with the model state_dict
            }

            dist_cp.load_state_dict(
                state_dict=state_dict, storage_reader=FileReader(resume_path)
            )

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FileReader(resume_path),
            )
            return state_dict["model"], optim_state["optim"]
