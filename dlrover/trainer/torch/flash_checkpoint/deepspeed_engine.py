import torch
import torch.distributed as dist
from datetime import timedelta

from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DeepSpeedCheckpointConfig,
    timer,
    DeepSpeedCheckpointSaver,
    CheckpointEvent,
    SharedMemoryHandler,
    CheckpointEventType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from .engine import CheckpointEngine


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
    """

    def __init__(self, checkpoint_dir, global_shard_num=1, zero_stage=0):
        self.global_shard_num = global_shard_num
        self.zero_stage = zero_stage
        super().__init__(checkpoint_dir)
        if dist.is_initialized():
            saver_ranks = self._get_saver_ranks()
            logger.info(f"Saver ranks of DeepSpeed is {saver_ranks}")
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
    def save_to_memory(
        self, step, state_dict, model_path="", optimizer_path=""
    ):
        """
        Synchronously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            model_path (str): the storage path to save the model state dict.
            optimizer_path (str): the storage path to save the optimizer
                state dict.
        """
        conf = DeepSpeedCheckpointConfig(
            step=step,
            model_path=model_path,
            optimizer_path=optimizer_path,
        )
        self._save_state_dict_to_memory(state_dict, conf)

    @timer
    def save_to_storage(
        self, step, state_dict, model_path="", optimizer_path=""
    ):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            model_path (str): the storage path to save the model state dict.
            optimizer_path (str): the storage path to save the optimizer
                state dict.
        """
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, model_path, optimizer_path)

        # Only local rank 0 to notify the saving event to the agent.
        if self._local_rank != 0:
            return
        if model_path or optimizer_path:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)

    def get_local_shard_num(self):
        local_world_size = env_utils.get_local_world_size()
        global_shard_num = self.get_global_shard_num()
        return min(local_world_size, global_shard_num)

    def get_global_shard_num(self):
        return self.global_shard_num

    def get_saver_class(self):
        return DeepSpeedCheckpointSaver

    def load(self, resume_model_path="", resume_optimizer_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self._shm_handler.load_state_dict()
        msd_name = CheckpointConstant.MODEL_STATES_NAME
        if msd_name not in state_dict and self.zero_stage in [1, 2]:
            local_rank_0_shm_handler = SharedMemoryHandler(0, host=False)
            # For stage 1,2, the model is not partitioned and only local rank 0
            # saves the model state dict into the CPU memory. Other local ranks
            # need get the model state dict from the shared memory of local
            # rank 0.
            sd = local_rank_0_shm_handler.load_state_dict()
            state_dict[msd_name] = sd[msd_name]
        if state_dict:
            return state_dict
        state_dict = self._load_from_storage(
            resume_model_path, resume_optimizer_path
        )
        return state_dict

    def _load_from_storage(
        self, resume_model_path="", resume_optimizer_path=""
    ):
        """
        Load the DeepSpeedEngine state dict from the storage.

        Args:
            resume_path (str, optional): , If the resume_path is an empty
                string, the function will load the latest checkpoint file in
                the checkpoint directory.

        Returns:
            A dict:
                a dictionary containing a whole state of the modules in the
                checkpointing file.
        """
        ds_state_dict = {}
        if resume_model_path:
            sd = torch.load(resume_model_path, map_location="cpu")
            ds_state_dict[CheckpointConstant.MODEL_STATES_NAME] = sd
        if resume_optimizer_path:
            sd = torch.load(resume_model_path, map_location="cpu")
            ds_state_dict[CheckpointConstant.OPTIM_STATES_NAME] = sd
        return ds_state_dict
