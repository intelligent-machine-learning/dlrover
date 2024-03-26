import os
import pickle
from typing import Dict, Optional

import torch
import torch.distributed as dist
from dlrover.python.common import env_utils
from dlrover.python.common.singleton import singleton
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    TempDirCheckpointSaver,
)
from dlrover.trainer.torch.flash_checkpoint.checkpointer import StorageType
from dlrover.trainer.torch.flash_checkpoint.engine import CheckpointEngine, timer

from .fsdp_save_util import get_flat_model_param, get_fsdp_optim_param, parallel_group, safetensors_dump

_PARAMS = "params"
_BUFFERS = "buffers"
_PARAM_META = "param_meta"
_CKPT_META = "ckpt_meta"
_OPTIM_STATES = "optim_states"
_PARAM_GROUPS = "param_groups"


class FsdpCheckpointSaver(TempDirCheckpointSaver):

    tracker_file = "latest_checkpointed_iteration.txt"

    def persist_to_storage(self, local_shard_id, ckpt_config: CheckpointConfig):
        """
        Persist the state dict to a storage path.

        Args:
            local_shard_id (int): the index of local shard.
            ckpt_config : the checkpoint config with the path to
                save the storage.
        """
        state_dict = self._shm_handlers[local_shard_id].load_state_dict()
        for name, path in ckpt_config.paths.items():
            state = state_dict.get(name, None)
            if not state:
                continue
            if name in [_PARAMS, _OPTIM_STATES, _BUFFERS]:
                safetensors_dump(state, path)
            elif name in [_PARAM_GROUPS, _PARAM_META, _CKPT_META]:
                with open(path, "wb") as f:
                    pickle.dump(state, f)
            else:
                torch.save(state_dict[name], path)

    def update_tracker_file(self, step):
        """
        Write the step into the tracker file.

        Args:
            step (int): the checkpointing step.
        """
        tracker_filename = os.path.join(self.checkpoint_dir, self.tracker_file)
        self.storage.write(str(step), tracker_filename)


@singleton
class FsdpCheckpointEngine(CheckpointEngine):
    def __init__(self, checkpoint_dir: str, storage=None, comm_backend=""):
        if storage is None:
            storage = PosixDiskStorage()
        super().__init__(checkpoint_dir, storage, comm_backend)

    def get_saving_ranks(self):
        """Get the ranks to save checkpoint shards."""
        return None

    @timer
    def save_to_memory(self, step, state_dict, paths):
        """
        Synchronously Save the state dict into the shared memory. If the agent in the main process
        is saving the checkpoint in the shared memory to the storage, the method will
        skip writing the shared memory. Only local rank 0 saves the state dict to the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a module name and the value is a path
                to save the module state dict.
        """
        conf = CheckpointConfig(step=step, paths=paths)
        return self.save_state_dict_to_memory(state_dict, conf)

    @timer
    def save_to_storage(self, step, state_dict, paths):
        """
        Asynchronously save the state dict into the storage. It firstly synchronously
        saves the state dict into the shared memory and put the path
        into a shared queue with the agent. Then, the agent in the main process saves the state dict
        in the shared memory to the storage. Only rank 0 sends a event to the agent to save
        the state dict to the storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a module name and the value is the path
                to save the module state dict.
        """
        succeed = True
        if step > self._cached_step:
            succeed = self.save_to_memory(step, state_dict, paths)

        # barrier to wait all ranks save state dict to memory.
        if dist.is_initialized():
            dist.barrier()
        # Only local rank 0 on each node notifies the saving event to the agent.
        if self._local_rank == 0 and succeed:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)

    def get_global_shard_num(self):
        if dist.is_initialized():
            return dist.get_world_size()
        return 0

    def get_local_shard_num(self):
        return env_utils.get_local_world_size()

    def load(self, resume_path=""):
        pass

    def get_saver_class(self):
        return FsdpCheckpointSaver


def save_checkpoint(
    step,
    model,
    optimizer,
    path,
    extra_sds: Optional[Dict] = None,
    extra_paths: Optional[Dict] = None,
    storage_type=StorageType.DISK,
    comm_backend="",
    storage=None,
):
    """
    Asynchronously save a checkpoint.

    Args:
        step (int): the iteration step.
        model: A FSDP model.
        optimizer: A optimizer initialized by a FSDP model.
        extra_sds: A dict where the key is a unique str and the value is a state dict to save using torch.save.
            For example: extra_sds["scheduler"]=scheduler.state_dict().
        extra_paths: A dict where the key is a unique str in the extra_sds
            and the value is a path to save. For example, extra_sds["scheduler"]="/tmp/scheduler.pt".
    """
    dir_name = os.path.dirname(path)
    ckpt_engine = FsdpCheckpointEngine(dir_name, storage=storage, comm_backend=comm_backend)
    params, buffers, param_meta, ckpt_meta = get_flat_model_param(model)
    optim_state, param_groups = get_fsdp_optim_param(model, optimizer)

    data_group = parallel_group("data")
    state_dict = {
        _PARAMS: params,
        _PARAM_META: param_meta,
        _OPTIM_STATES: optim_state,
    }

    if dist.get_rank(data_group) == 0:
        state_dict[_PARAM_GROUPS] = param_groups
        state_dict[_BUFFERS] = buffers

    if dist.get_rank() == 0:
        state_dict[_CKPT_META] = ckpt_meta

    if extra_sds:
        state_dict.update(extra_sds)
    paths = {}
    model_paths = _get_model_ckpt_paths(path)
    paths.update(model_paths)
    optim_paths = _get_optim_ckpt_paths(path)
    paths.update(optim_paths)
    if extra_paths:
        paths.update(extra_paths)
    if storage_type == StorageType.MEMORY:
        ckpt_engine.save_to_memory(step, state_dict, paths)
    elif storage_type == StorageType.DISK:
        ckpt_engine.save_to_storage(step, state_dict, paths)
    else:
        raise ValueError("The storage type only supports StorageType.MEMORY and StorageType.DISK")


def _get_model_ckpt_paths(path):
    """Persit the flat model parameters into the storage."""
    data_group = parallel_group("data")
    suffix = f"{str(dist.get_rank(data_group)).zfill(5)}-{str(dist.get_world_size(data_group)).zfill(5)}"
    paths: Dict[str, str] = {}
    paths[_PARAMS] = f"{path}/flat_param.{suffix}"
    paths[_BUFFERS] = f"{path}/buffers"
    paths[_PARAM_META] = f"{path}/flat_meta.{suffix}"
    paths[_CKPT_META] = f"{path}/ckpt_meta"
    return paths


def _get_optim_ckpt_paths(path):
    """Persit the optimizer parameters into the storage."""
    data_group = parallel_group("data")
    suffix = f"{str(dist.get_rank(data_group)).zfill(5)}-{str(dist.get_world_size(data_group)).zfill(5)}"
    paths: Dict[str, str] = {}
    paths[_OPTIM_STATES] = f"{path}/optim_param.{suffix}"
    paths[_PARAM_GROUPS] = f"{path}/optim_meta"
    return paths
