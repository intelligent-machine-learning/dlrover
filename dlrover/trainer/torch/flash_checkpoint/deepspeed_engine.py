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

import copy

import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    DeepSpeedCheckpointSaver,
    SharedMemoryHandler,
)

from .engine import CheckpointEngine, timer


class DeepSpeedCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict of
    `DeepSpeedEngine` into the shared memory and notify the agent
    in main process to asynchronously save the state dict from the shared
    memory into the storage.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.
        dp_size (int): the world size of data parallelism.
        global_shard_num (int): the number of shards across all ranks.
        zero_stage (int): the DeepSpeed ZERO Stage number.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
    """

    def __init__(
        self,
        checkpoint_dir,
        storage,
        global_shard_num=1,
        zero_stage=0,
        comm_backend="",
        save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    ):
        self.global_shard_num = global_shard_num
        self.zero_stage = zero_stage
        super().__init__(checkpoint_dir, storage, comm_backend, save_timeout)

    def get_saving_ranks(self):
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
    def save_to_memory(self, step, state_dict, paths):
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
        success = self.save_state_dict_to_memory(state_dict, conf)
        return success

    @timer
    def save_to_storage(self, step, state_dict, paths):
        """
        Asynchonously saves the state dict into the storage. It synchonously
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
        success = True
        if step > self._cached_step:
            success = self.save_to_memory(step, state_dict, paths)

        if dist.is_initialized():
            dist.barrier()

        # Only local rank 0 to notify the saving event to the agent.
        if self._local_rank == 0 and success:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)
        return success

    def get_local_shard_num(self):
        local_world_size = env_utils.get_local_world_size()
        global_shard_num = self.get_global_shard_num()
        return min(local_world_size, global_shard_num)

    def get_global_shard_num(self):
        return self.global_shard_num

    def get_saver_class(self):
        return DeepSpeedCheckpointSaver

    def load(self):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        _, state_dict = self.get_state_dict_from_memory()
        if state_dict:
            msd_name = CheckpointConstant.MODEL_STATES_NAME
            if msd_name not in state_dict and self.zero_stage in [1, 2]:
                local_rank_0_shm_handler = SharedMemoryHandler(0, host=False)
                # For stage 1,2, the model is not partitioned and only local
                # rank 0 saves the model state dict into the CPU memory. Other
                # local ranks need get the model state dict from the shared
                # memory of local rank 0.
                sd = local_rank_0_shm_handler.load_state_dict()

                # Deep copy the model state dict because a state dict only can
                # be loaded by one rank.
                state_dict[msd_name] = copy.deepcopy(sd[msd_name])
        return state_dict
