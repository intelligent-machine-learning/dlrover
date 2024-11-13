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
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import SharedLock, SharedQueue
from dlrover.python.common.singleton import Singleton
from dlrover.python.common.storage import CheckpointStorage
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    AsyncCheckpointSaver,
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    CheckpointSharedObjPrefix,
    ClassMeta,
    SharedMemoryHandler,
)
from dlrover.trainer.torch.flash_checkpoint.replica import CkptReplicaManger


def _local_rank0_log(local_rank, message):
    if local_rank == 0:
        logger.info(message)


class ReadyTensor(Singleton):
    def __init__(self, device) -> None:
        self.tensor = torch.tensor([0], dtype=torch.int32).to(device)


def check_all_rank_ready(group: dist.ProcessGroup, ready: bool):
    """
    Check whether all ranks are ready.
    """
    if not group and not dist.is_initialized():
        return ready
    backend = dist.get_backend(group)
    local_rank = env_utils.get_local_rank()
    device = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    rt = ReadyTensor.singleton_instance(device)
    value = 0 if ready else 1
    rt.tensor[0] = value
    dist.all_reduce(rt.tensor, group=group)
    ready = rt.tensor == 0
    return ready


def verify_all_rank_step_consistent(group: dist.ProcessGroup, step):
    """
    Verify whether the step in all ranks are consistent.
    """
    if not group and not dist.is_initialized():
        return True
    backend = dist.get_backend(group)
    local_rank = env_utils.get_local_rank()
    device = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    t = torch.tensor([float(step)]).to(device)
    if group:
        world_size = group.size()
    else:
        world_size = dist.get_world_size()
    outputs = [torch.tensor([0.0]).to(device) for _ in range(world_size)]
    dist.all_gather(outputs, t, group=group)
    succeed = True
    for step in outputs:
        if not torch.equal(step, outputs[0]):
            succeed = False
    del t, outputs
    return succeed


def timer(func):
    def wrapper(*args, **kwargs):
        local_rank = env_utils.get_local_rank()
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(
            f"Local rank {local_rank } execute {func.__name__} in {t}s."
        )
        return result

    return wrapper


def start_async_save():
    AsyncCheckpointSaver.start_async_saving_ckpt()
    while True:
        time.sleep(60)


def start_saver_process():
    """
    Start a process to asynchronously save checkpoint if the training
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
        storage: a CheckpointStorage instance to write/read the storage.
        comm_backend (str): the communication backend to create a process
            group, default: backend of general main process group.
    """

    saver_proc = None

    def __init__(
        self,
        checkpoint_dir: str,
        storage: CheckpointStorage,
        comm_backend: str = "",
        save_timeout: int = CheckpointConstant.SAVE_TIMEOUT,
        replica_count=0,
    ):
        logger.info(
            "Initializing checkpoint engine: "
            f"{self.__class__.__name__.lower()}."
        )
        if not self.saver_proc:
            self.saver_proc = start_saver_process()

        self.checkpoint_dir = checkpoint_dir
        self.storage = storage
        self._save_timeout = save_timeout
        self._local_rank = env_utils.get_local_rank()
        self._cached_step = -1
        self._restart_count = env_utils.get_torch_restart_count()

        # init saver
        self._notify_agent_to_create_saver()

        # queue for agent to save to storage, only lock rank 0 needs the queue.
        if self._local_rank == 0:
            self._event_queue = SharedQueue(
                name=CheckpointSharedObjPrefix.SAVE_STEP_QNAME + str(0),
                create=False,
            )
        else:
            self._event_queue = None  # type: ignore
        self._update_saver_config()

        # lock for shared memory
        local_shard_num = self.get_local_shard_num()
        self.local_shard_id = self._local_rank % local_shard_num
        lock_name = CheckpointSharedObjPrefix.SHM_LOCK_NAME + str(
            self.local_shard_id
        )
        self._shm_lock = SharedLock(name=lock_name, create=False)

        # need to wait until the socket server is created(by the saver)
        while not self._shm_lock.is_available():
            time.sleep(0.1)

        self._shm_handler = SharedMemoryHandler(
            self.local_shard_id, host=False
        )
        self._rank = 0
        self._group_rank = 0
        self._world_size = 1
        self._loader_group = None
        self._saver_group = None
        self._saving_ranks: Optional[List[int]] = None
        self._init_sync_group(comm_backend)
        shard_num = self.get_global_shard_num()
        self._replica_manager = CkptReplicaManger.create_replica_manager(
            shard_num, replica_count
        )
        logger.info(
            "Checkpoint engine initialized with "
            f"local rank: {self._local_rank}, rank: {self._rank}."
        )

    def _init_sync_group(self, comm_backend):
        if not dist.is_initialized():
            self._saving_ranks = [0]
            return

        self._rank = dist.get_rank()
        self._group_rank = env_utils.get_group_rank()
        self._world_size = dist.get_world_size()
        backend = comm_backend if comm_backend else dist.get_backend()
        if backend == dist.get_backend():
            self._loader_group = None
        else:
            self._loader_group = dist.new_group(
                backend=backend,
                timeout=timedelta(seconds=60),
            )
        self._saving_ranks = self.get_saving_ranks()
        if backend == dist.get_backend() and (
            self._saving_ranks is None
            or len(self._saving_ranks) == dist.get_world_size()
        ):
            self._saver_group = None
            message = (
                "Use the default process group to sync "
                "when saving checkpoint."
            )
            _local_rank0_log(self._local_rank, message)
        else:
            self._saver_group = dist.new_group(
                ranks=self._saving_ranks,
                backend=backend,
                timeout=timedelta(seconds=60),
            )
            if self._saving_ranks:
                message = (
                    f"Create a {backend} communication group to save "
                    f"checkpoint. Saving ranks are {self._saving_ranks}."
                )
            else:
                message = (
                    f"Create a {backend} communication group to save "
                    "checkpoint. Saving ranks are all ranks."
                )
            _local_rank0_log(self._local_rank, message)

    def __del__(self):
        self.close()

    def close(self):
        """Close the shared memory."""
        self._shm_handler.close()

    def _notify_agent_to_create_saver(self):
        """Notify the agent in the main process to create a checkpoint saver"""
        if self._local_rank != 0:
            return

        # the agent side will release the lock if training process restarts.
        queue = SharedQueue(name="factory")

        local_shard_num = self.get_local_shard_num()
        global_shard_num = self.get_global_shard_num()
        clazz = self.get_saver_class()
        class_meta = ClassMeta(
            module_path=clazz.__module__,
            class_name=clazz.__name__,
            kwargs={
                "checkpoint_dir": self.checkpoint_dir,
                "storage_meta": self.storage.get_class_meta(),
                "local_shard_num": local_shard_num,
                "global_shard_num": global_shard_num,
                "save_timeout": self._save_timeout,
            },
        )

        logger.info(
            "Notify agent to create a checkpoint saver using: "
            f"{class_meta.__dict__}."
        )
        queue.put(class_meta)

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

            while not self._event_queue.is_available():
                time.sleep(0.1)

            logger.info(f"Update saver config: {event.__dict__}")
            self._event_queue.put(event)

    def save_state_dict_to_memory(self, state_dict, conf: CheckpointConfig):
        """Save the state dict into the memory."""
        if self._local_rank != self.local_shard_id:
            return False
        if self._saving_ranks and self._rank not in self._saving_ranks:
            return False

        conf.rank = self._rank
        conf.group_rank = self._group_rank
        conf.world_size = self._world_size

        acquired = self._shm_lock.acquire(blocking=False)
        logger.info(
            f"{self._rank}-{self._local_rank} acquired the lock of shared "
            f"memory: {acquired} for step: {conf.step}."
        )
        all_rank_ready = check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready or not state_dict:
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
        self._replica_manager.backup(self._shm_handler)
        return True

    def get_state_dict_from_memory(self):
        """
        Restore the checkpoint state dict from the shared memory.
        """
        self._restore_memory_from_replica()
        state_dict = {}
        default_config = CheckpointConfig()
        config = self._shm_handler.get_checkpoint_config(default_config)
        passed = verify_all_rank_step_consistent(
            self._loader_group, config.step
        )
        if passed and config.step > 0:
            state_dict = self._shm_handler.load_state_dict()
            state_dict.pop(DLROVER_CKPT_CONFIG_KEY, None)
            logger.info(f"Load checkpoint at step {config.step} from memory.")
        return config.step, state_dict

    def _restore_memory_from_replica(self):
        if not self._replica_manager.has_replica():
            return
        self._shm_handler.init_shared_memory()
        byte_tensor, meta = self._replica_manager.gather(self._shm_handler)
        if (
            byte_tensor is not None
            and meta
            and not self._shm_handler.shared_memory
        ):
            shm_size = byte_tensor.size()[0]
            self._shm_handler.init_shared_memory(create=True, size=shm_size)
            self._shm_handler.metadata.set(meta)
            logger.info(
                f"Restore the checkpoint shard with size = {shm_size}"
                "from the replica in the memory of the alive node."
            )
        dist.barrier()

    @abstractmethod
    def get_saving_ranks(self):
        pass

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
        Asynchronously save the state dict into the storage. It firstly
        synchronously saves the state dict into the shared memory and
        put the path into a shared queue with the agent. Then, the agent
        in the main process saves the state dict in the shared memory to the
        storage. Only rank 0 sends a event to the agent to save
        the state dict to the storage.

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
