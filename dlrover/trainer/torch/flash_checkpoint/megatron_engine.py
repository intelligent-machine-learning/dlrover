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

from datetime import timedelta

import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointEvent,
    CheckpointEventType,
    MegatronCheckpointSaver,
)

from .engine import CheckpointEngine, timer


class MegatronCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict of
    Megatron-LM model and optimizer into
    the shared memory and notify the agent in main process to
    asynchronously save the state dict from the shared memory into
    the storage.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.
    """

    def __init__(self, checkpoint_dir):
        if dist.is_initialized():
            from megatron import mpu

            self._tp_rank = mpu.get_tensor_model_parallel_rank()
            self._pp_rank = mpu.get_pipeline_model_parallel_rank()
            self._pp_world_size = mpu.get_pipeline_model_parallel_world_size()
            self._tp_world_size = mpu.get_tensor_model_parallel_world_size()
            self._dp_rank = mpu.get_data_parallel_rank()
        else:
            self._tp_rank = 0
            self._pp_rank = 0
            self._dp_rank = 0
            self._pp_world_size = 1
            self._tp_world_size = 1

        super().__init__(checkpoint_dir)
        if dist.is_initialized():
            saver_ranks = self._get_saver_ranks()
            logger.info(f"Saver ranks of Megatron-LM is {saver_ranks}")
            self._saver_group = dist.new_group(
                ranks=saver_ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )

    def _get_saver_ranks(self):
        """
        Get the ranks which need to save the sharding state dict into
        the memory.
        """
        world_size = dist.get_world_size()
        local_world_size = env_utils.get_local_world_size()
        save_ranks = []
        local_shard_num = self.get_local_shard_num()
        for i in range(world_size):
            local_rank = i % local_world_size
            if local_rank < local_shard_num:
                save_ranks.append(i)
        return save_ranks

    @timer
    def save_to_storage(self, step, state_dict, path):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
        """
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, path)

        # Only local rank 0 to notify the saving event to the agent.
        if self._dp_rank != 0 or self._local_rank != 0:
            return
        if path:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)

    def get_local_shard_num(self):
        local_world_size = env_utils.get_local_world_size()
        global_shard_num = self.get_global_shard_num()
        return min(local_world_size, global_shard_num)

    def get_global_shard_num(self):
        num = self._pp_world_size * self._tp_world_size
        return num

    def get_saver_class(self):
        return MegatronCheckpointSaver

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
            return state_dict
        state_dict = self._load_from_storage(resume_path)
        return state_dict

    def _load_from_storage(self, resume_path=""):
        """
        Load the state dict from the storage.
        Args:
            resume_path (str, optional): , If the resume_path is an empty
                string, the function will load the latest checkpoint file in
                the checkpoint directory.

        Returns:
            A dict:
                a dictionary containing a whole state of the modules in the
                checkpointing file.
        """
        if resume_path:
            from torch.serialization import load as torch_load

            state_dict = torch_load(resume_path, map_location="cpu")
            return state_dict
        return {}
