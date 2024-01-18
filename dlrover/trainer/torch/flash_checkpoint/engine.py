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
import time
from abc import ABCMeta, abstractmethod
from datetime import timedelta
from multiprocessing import Process
from typing import Dict

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import SharedLock, SharedQueue
from dlrover.python.common.storage import CheckpointStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    AsyncCheckpointSaver,
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    CheckpointSharedObjPrefix,
    SaverClassMeta,
    SharedMemoryHandler,
)


def check_all_rank_ready(group: dist.ProcessGroup, ready):
    """
    Check whether all ranks are ready.
    """
    if not group:
        return ready
    backend = dist.get_backend()
    local_rank = env_utils.get_local_rank()
    device = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    value = 0 if ready else 1
    t = torch.tensor([value], dtype=torch.int32).to(device)
    dist.all_reduce(t, group=group)
    ready = t == 0
    del t
    if "cuda" in device:
        torch.cuda.empty_cache()
    return ready


def verify_all_rank_step_consistent(group: dist.ProcessGroup, step):
    """
    Verify whether the step in all ranks are consistent.
    """
    if not group:
        return True
    backend = dist.get_backend()
    local_rank = env_utils.get_local_rank()
    device = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    t = torch.tensor([float(step)]).to(device)
    world_size = group.size()
    outputs = [torch.tensor([0.0]) for _ in range(world_size)]
    dist.all_gather(outputs, t, group=group)
    succeed = True
    for step in outputs:
        if not torch.equal(step, outputs[0]):
            succeed = False
    del t, outputs
    if "cuda" in device:
        torch.cuda.empty_cache()
    return succeed


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(f"Function {func.__name__} cost {t}s")
        return result

    return wrapper


def start_async_save():
    AsyncCheckpointSaver.start_async_saving_ckpt()
    while True:
        time.sleep(36000)


def start_saver_process():
    """
    Start a process to to asynchronously save checkpoint if the training
    process is not launched by `dlrover-run`. This process will
    exit and cannot save the checkpoint after the training process exit.
    It is better to use `dlrover-run` to start the training process.
    `dlrover-run` can save checkpoint once the training process fails
    and relaunch new training processes which can restore the checkpoint
    from the memory not the storage.
    """
    local_rank = env_utils.get_local_rank()
    role_name = os.getenv("ROLE_NAME", "")
    # Only start the process on local rank 0
    # if the training process is not launched by dlrover-run.
    if role_name != "dlrover-trainer" and local_rank == 0:
        p = Process(target=start_async_save, daemon=True)
        p.start()
        logger.info("Start a process to asynchronously save checkpoint.")
        return p
    return None


class CheckpointEngine(metaclass=ABCMeta):
    """
    The checkpoint engine synchronously writes the state dict into
    the shared memory and notify the agent in main process to
    asynchronously save the state dict from the shared memory into
    the storage. Writing to memory is significantly quicker
    than writing to storage. The engine only blocks the training
    with a little time. Users can frequently call `save_to_memory` in
    the training loop and call `save_to_storage`.

    If the training process fail, the agent in main process can continuously
    saves the state dict from the shared memory into the storage.

    Args:
        checkpoint_dir (str): the directory to save checkpoint.
    """

    saver_proc = None

    def __init__(self, checkpoint_dir: str, storage: CheckpointStorage):
        if not self.saver_proc:
            self.saver_proc = start_saver_process()
        self.checkpoint_dir = checkpoint_dir
        self.storage = storage
        if dist.is_initialized():
            self._rank = dist.get_rank()
            backend = dist.get_backend()
            self._loader_group = dist.new_group(
                backend=backend,
                timeout=timedelta(seconds=30),
            )
        else:
            self._rank = 0
            self._loader_group = None
        self._local_rank = int(os.getenv("LOCAL_RANK", 0))
        self._saver_group = None
        self._cached_step = 0
        self._restart_count = env_utils.get_torch_restart_count()
        # queue for agent to save to storage, only lock rank 0 needs the queue.
        if self._local_rank == 0:
            self._event_queue = SharedQueue(
                name=CheckpointSharedObjPrefix.SAVE_STEP_QNAME + str(0),
                create=False,
            )
        else:
            self._event_queue = None  # type: ignore
        # lock for shared memory
        local_shard_num = self.get_local_shard_num()
        self.local_shard_id = self._local_rank % local_shard_num
        lock_name = CheckpointSharedObjPrefix.SHM_LOCK_NAME + str(
            self.local_shard_id
        )
        self._shm_lock = SharedLock(name=lock_name, create=False)
        self._shm_handler = SharedMemoryHandler(
            self.local_shard_id, host=False
        )
        self._notify_agent_to_create_saver()
        self._update_saver_config()

    def __del__(self):
        self.close()

    def close(self):
        """Close the shared memory."""
        self._shm_handler.close()

    def _notify_agent_to_create_saver(self):
        """Notify the agent in the main process to create a checkpoint saver"""
        if self._local_rank != 0:
            return
        if self._restart_count > 0:
            # Only local rank 0 notify to initialize the saver in
            # the main process at the first start.
            # Avoid the lock is locked by a failed process.
            self._shm_lock.release()
            return
        queue = SharedQueue(name="factory")

        local_shard_num = self.get_local_shard_num()
        global_shard_num = self.get_global_shard_num()
        clazz = self.get_saver_class()
        class_meta = SaverClassMeta(
            module_path=clazz.__module__,
            class_name=clazz.__name__,
            init_args={
                "checkpoint_dir": self.checkpoint_dir,
                "storage": self.storage,
                "local_shard_num": local_shard_num,
                "global_shard_num": global_shard_num,
            },
        )

        queue.put(class_meta)
        queue.unlink()

    def _update_saver_config(self):
        """Update the sharding configuration to the saver."""
        if self._local_rank == 0:
            global_shard_num = self.get_global_shard_num()
            event: CheckpointEvent = CheckpointEvent(
                type=CheckpointEventType.UPDATE_SHARD,
                global_shard_num=global_shard_num,
            )
            if self._event_queue is None:
                raise ValueError(
                    "The event queue cannot be None on local rank 0."
                )
            self._event_queue.put(event)

    def save_state_dict_to_memory(self, state_dict, conf: CheckpointConfig):
        if self._local_rank != self.local_shard_id:
            return False

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
            return False
        state_dict[DLROVER_CKPT_CONFIG_KEY] = conf
        self._shm_handler.save_state_dict(state_dict)

        if acquired:
            self._shm_lock.release()
        self._cached_step = conf.step
        if dist.is_initialized():
            dist.barrier(group=self._saver_group)
        return True

    def get_state_dict_from_memory(self):
        state_dict = {}
        default_config = CheckpointConfig()
        config = self._shm_handler.get_checkpoint_config(default_config)
        passed = verify_all_rank_step_consistent(
            self._loader_group, config.step
        )
        if passed and config.step > 0:
            state_dict = self._shm_handler.load_state_dict()
            state_dict.pop(DLROVER_CKPT_CONFIG_KEY, None)
            logger.info(
                f"Load step {config.step} checkpoint from the shared memory."
            )
        return state_dict

    @abstractmethod
    def get_saver_class(self):
        """
        Get a CheckpointSaver class.
        """
        pass

    @abstractmethod
    def get_local_shard_num(self):
        """Get the number of model shards on the node."""
        pass

    @abstractmethod
    def get_global_shard_num(self):
        """Get the number of model shards on all nodes."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def save_to_storage(self, step, state_dict, paths: Dict[str, str]):
        """
        Save the state_dict into the path of storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a category in
                ["model_states", "optim_states"] of the state dict and
                the value is the path of storage to save.
        """
        pass

    @abstractmethod
    def load(self, resume_path=""):
        """
        Load the checkpointing state dict from the resume path.

        Returns:
            A dict.
        """
        pass
