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

import importlib
import os
import shutil
import signal
import threading
import time
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import (
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
)

_CKPT_DIR_PREFIX = "checkpoint-"

_SAVE_STEP_QNAME_PREFIX = "checkpoint_lock_rank_"
_CKPT_META_NAME_PREFIX = "checkpoint_meta_local_rank_"
_TENSOR_SHM_NAME_PREFIX = "checkpoint_shm_local_rank_"
_SHM_LOCK_NAME_PREFIX = "shm_local_rank_"
_WIRTING_SHM = "__WRITING_SHM__"


@dataclass
class SaverClassMeta:
    module_path: str = ""
    class_name: str = ""
    init_args: Dict[str, str] = None  # type: ignore


@dataclass
class TensorMeta(object):
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(f"Function {func.__name__} cost {t}s")
        return result

    return wrapper


def _init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def _traverse_state_dict(value: object, visitor: Callable[[object], None]):
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    """
    if isinstance(value, Mapping):
        temp_dict = {}
        for k, v in value.items():
            temp_dict[k] = _traverse_state_dict(v, visitor)
        return temp_dict
    elif isinstance(value, List):
        temp_list = []
        for _, v in enumerate(value):
            temp_list.append(_traverse_state_dict(v, visitor))
        return temp_list
    else:
        return visitor(value)


def _read_state_dict_from_shm(meta_dict, tensor_shm):
    state_dict = _traverse_state_dict(
        meta_dict,
        lambda x: _read_tensor_from_buf(x, tensor_shm),
    )
    return state_dict


def _read_tensor_from_buf(value, shm_tensor_buffer):
    """
    Read a tensor from the buffer of shared memory.
    """
    if isinstance(value, TensorMeta):
        shm_tensor = torch.frombuffer(
            buffer=shm_tensor_buffer.buf,
            dtype=value.dtype,
            offset=value.offset,
            count=value.numel,
        )
        value = shm_tensor.reshape(value.shape)
        return value
    else:
        return value


def _create_shared_memory(name, create, size=0):
    """
    Create a shared memory.
    """
    if not create:
        try:
            return SharedMemory(name=name)
        except FileNotFoundError:
            return None
    try:
        shm = SharedMemory(
            name=name,
            create=create,
            size=size,
        )
    except FileExistsError:
        shm = SharedMemory(name=name)
    return shm


def _get_latest_checkpoint(checkpoint_dir):
    """Get the checkpoint directory with the maximum step."""
    if not os.path.exists(checkpoint_dir):
        return ""
    max_step = 0
    for fn in os.listdir(checkpoint_dir):
        if not fn.startswith(_CKPT_DIR_PREFIX):
            continue
        step = int(fn.split("-")[-1])
        max_step = step if step > max_step else max_step
    if max_step > 0:
        path = os.path.join(checkpoint_dir, f"{_CKPT_DIR_PREFIX}{max_step}")
    else:
        path = ""
    return path


def _check_all_rank_ready(group, ready):
    """
    Check wether all ranks are ready.
    """
    if not group:
        return ready
    value = 0 if ready else 1
    t = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(t, group=group)
    return t == 0


def _tarverse_copy_to_shm(value, meta, buffer):
    if isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(v, (Mapping, List)):
                m = meta[k]
                _tarverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                m = meta[k]
                _write_shared_memory(v, m, buffer)
            else:
                meta[k] = v
    elif isinstance(value, List):
        for i, v in enumerate(value):
            if isinstance(v, (Mapping, List)):
                m = meta[i]
                _tarverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                m = meta[i]
                _write_shared_memory(v, m, buffer)
            else:
                meta[i] = v


def _write_shared_memory(value: torch.Tensor, meta: TensorMeta, buffer):
    """
    Write a CPU tensor into the shared memory.
    """
    shm_tensor = torch.frombuffer(
        buffer, dtype=value.dtype, count=value.numel(), offset=meta.offset
    ).reshape(value.shape)
    shm_tensor.copy_(value)


def _load_from_historic_checkpoint(checkpoint_dir):
    """Locd checkpoint from the lastest complete checkpoint."""
    while True:
        latest_ckpt_dir = _get_latest_checkpoint(checkpoint_dir)
        if not latest_ckpt_dir:
            return {}

        resume_path = os.path.join(latest_ckpt_dir, "checkpoint.pt")
        if not os.path.exists(resume_path):
            shutil.rmtree(latest_ckpt_dir)
            continue
        try:
            state_dict = torch.load(resume_path)
            logger.info(f"Load checkpoint from {resume_path}")
            return state_dict
        except Exception:
            logger.warning(
                f"Fail to load checkpoint from {resume_path}."
                " Roll back to the last checkpoint file."
            )
            shutil.rmtree(latest_ckpt_dir)


class CheckpointSaver(metaclass=ABCMeta):
    """
    CheckpointSaver saves the state dict from the shared memory into
    the storage.

    Attributes:
        checkpoint_dir (str): the directory to save the checkpointing state
            dict to the storage if the training process fails.
        num_shard (int): the number of param sharding.
    """

    _saver_instance = None

    def __init__(self, checkpoint_dir, num_shard=1):
        self.checkpoint_dir = checkpoint_dir
        self.num_shard = num_shard

    @classmethod
    def start_async_saving_ckpt(cls):
        """
        Start a thread to asynchronously save the checkpoint state dict
        from the shared memory into the storage. Firstly, it waits that
        the training process notify the saver class to create a saver.
        """
        sq = SharedQueue(name="factory", create=True)

        def _save():
            class_meta: SaverClassMeta = sq.get()
            module = importlib.import_module(class_meta.module_path)
            class_def = getattr(module, class_meta.class_name)
            if cls._saver_instance is None:
                saver: CheckpointSaver = class_def(**class_meta.init_args)
                cls._saver_instance = saver
            cls._saver_instance._sync_shm_to_storage()

        threading.Thread(
            target=_save, name="checkpoint-saver", daemon=True
        ).start()

    @abstractmethod
    def _sync_shm_to_storage(self):
        pass

    @classmethod
    def get_ckpt_saver(cls):
        return cls._saver_instance

    @classmethod
    def register_signal_handler(cls):
        sigint_handler = signal.getsignal(signal.SIGINT)
        sigterm_handler = signal.getsignal(signal.SIGTERM)

        def _clean_shm_handler(signum, frame):
            """Clean the shared memory from ^C and "killall python" etc."""
            if cls._saver_instance:
                cls._saver_instance.close()
            if callable(sigint_handler):
                sigint_handler(signum, frame)

        def _save_shm_before_exiting(signum, frame):
            """Save the state dict from the shared memory into the storage
            before the process exits.
            """
            if cls._saver_instance:
                cls._saver_instance.save_shm_to_storage()
                cls._saver_instance.close()
            if callable(sigterm_handler):
                sigterm_handler(signum, frame)

        signal.signal(signal.SIGINT, _clean_shm_handler)
        signal.signal(signal.SIGTERM, _save_shm_before_exiting)

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save_shm_to_storage(self):
        pass

    def __del__(self):
        self.close()


class NoShardingSaver(CheckpointSaver):
    """
    The saver only saves the state dict without sharding
    from the shared memory created by local rank 0 to the storage.
    """

    def __init__(self, checkpoint_dir, num_shard=1) -> None:
        super().__init__(checkpoint_dir, num_shard)
        self._tensor_shm = None
        # Only local rank 0 save the state dict to memory in DDP.
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=True)
        meta_name = _CKPT_META_NAME_PREFIX + str(0)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=True)
        lock_name = _SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=True)
        self._shm_name = _TENSOR_SHM_NAME_PREFIX + str(0)

    def close(self):
        if self._tensor_shm:
            self._tensor_shm.close()
            self._tensor_shm.unlink()
        self._to_save_queue.close()
        self._shared_ckpt_meta.close()
        self._shm_lock.close()

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Async checkpoint saver starts!")
        while True:
            path = self._to_save_queue.get()
            if not self._tensor_shm:
                self._tensor_shm = SharedMemory(name=self._shm_name)
            self._shm_lock.acquire()
            logger.info(
                "Save checkpoint from the shared memory "
                f"into the storage {path}."
            )
            meta_dict = self._shared_ckpt_meta.get()
            state_dict = _read_state_dict_from_shm(meta_dict, self._tensor_shm)
            self._persist_to_storage(state_dict, path)
            self._shm_lock.release()

    def _persist_to_storage(self, state_dict, path):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        checkpoint_dir = os.path.dirname(path)
        state_dict.pop(_WIRTING_SHM, None)
        _init_dir(checkpoint_dir)
        torch.save(state_dict, path)

    def save_shm_to_storage(self):
        """
        Save the state dict in the shared memory into the storage. The agent
        can call the method to save the state dict into the storage if the
        training process fails or the agent wants to restart training
        processes.
        """
        if self._tensor_shm is None:
            return
        acquired = self._shm_lock.acquire()
        if not acquired:
            # The training process does not release the lock because it fails
            # when writing the state dict into the shared memory. The shared
            # memory may be dirty and the saver cannot save it to the storage.
            return
        meta_dict = self._shared_ckpt_meta.get()
        step = meta_dict["step"]
        path = os.path.join(
            self.checkpoint_dir, f"checkpoint-{step}/checkpoint.pt"
        )
        state_dict = _read_state_dict_from_shm(meta_dict, self._tensor_shm)
        self._persist_to_storage(state_dict, path)
        self._shm_lock.release()
        logger.info(
            "Save the checkpointing state dict from the shared "
            f"memory to {path}."
        )


class ShardingSaver(CheckpointSaver, ABC):
    """
    This saver will save the state dict for all ranks, because the state
    dict is sharded across all ranks.
    """

    STAGE_DIR = "._dlrover_ckpt_stage"

    def __init__(self, checkpoint_dir, num_shard=1) -> None:
        super().__init__(checkpoint_dir, num_shard)
        self._tensor_shm: List[Optional[SharedMemory]] = [
            None for _ in range(num_shard)
        ]
        self._shared_ckpt_meta = []
        self._shm_lock = []
        self._shm_name = []

        # Only local rank 0 will notify the agent to save status to storage
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=True)

        # Each rank has a shared memory to store the state dict
        for i in range(num_shard):
            meta_name = _CKPT_META_NAME_PREFIX + str(i)
            self._shared_ckpt_meta.append(
                SharedDict(name=meta_name, create=True)
            )
            lock_name = _SHM_LOCK_NAME_PREFIX + str(i)
            self._shm_lock.append(SharedLock(name=lock_name, create=True))
            shm_name = _TENSOR_SHM_NAME_PREFIX + str(i)
            self._shm_name.append(shm_name)

        self._node_num = env_utils.get_node_num()
        self._node_rank = env_utils.get_node_rank()
        self._is_agent_rank_0 = self._node_rank == 0
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_shard, thread_name_prefix="ckpt_saver-"
        )

        self._writing_storage = False

    @abstractmethod
    def persist_to_storage(self, state_dict, path):
        """
        Persist the checkpoint from CPU memory buffer into the storage.
        """

    @abstractmethod
    def update_tracker_file(self, step):
        pass

    def get_ckpt_name(self, step):
        """User can override the method to define the checkpoint name."""
        return f"checkpoint-{step}"

    def _get_ckpt_path(self, step):
        return os.path.join(self.checkpoint_dir, self.get_ckpt_name(step))

    def _persist_to_storage(self, state_dict, path, step):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        # save to tmp dir for each local rank
        if state_dict["step"] != step:
            raise RuntimeError(
                f"state_dict step {state_dict['step']} != step {step}"
            )

        self.persist_to_storage(state_dict, path)

    def _get_stage_path(self):
        """Stage directory for the checkpointing state dict."""
        return os.path.join(self.checkpoint_dir, self.STAGE_DIR)

    def close(self):
        for shm in self._tensor_shm:
            if shm:
                shm.close()
                shm.unlink()

        self._to_save_queue.close()

        for d in self._shared_ckpt_meta:
            d.close()

        for lock in self._shm_lock:
            lock.close()

        self._executor.shutdown()

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info(
            "ShardingSaver Start saving the checkpointing state dict to "
            "storage."
        )

        while True:
            step = self._to_save_queue.get()
            logger.info(
                "ShardingSaver save checkpoint to storage, step: %s", step
            )
            self._save_shm_to_storage(step)

    def _save_shm_to_storage(self, step):
        """
        Save all the local state dict in the shared memory into the storage
        for step.
        """
        logger.info(
            f"Rank {self._node_rank} start save checkpoint to storage, "
            f"step: {step}"
        )
        self._writing_storage = True
        ckpt_path = self._get_ckpt_path(step)

        if os.path.exists(ckpt_path):
            logger.info(f"Checkpoint for step {step} already exists, skip")
            self._writing_storage = False
            return

        def _save_stage(local_rank: int, write_path: str):
            try:
                if not self._tensor_shm[local_rank]:
                    self._tensor_shm[local_rank] = SharedMemory(
                        name=self._shm_name[local_rank]
                    )

                self._shm_lock[local_rank].acquire()
                logger.info(
                    f"Local rank {local_rank} Save checkpoint from the shared "
                    f"memory into the storage {write_path}."
                )
                meta_dict = self._shared_ckpt_meta[local_rank].get()
                state_dict = _read_state_dict_from_shm(
                    meta_dict, self._tensor_shm[local_rank]
                )

                self._persist_to_storage(state_dict, write_path, step)
                self._shm_lock[local_rank].release()
                return True

            except Exception as e:
                logger.error(
                    f"Rank {local_rank} save checkpoint failed, error: {e}",
                    exc_info=True,
                )
                self._shm_lock[local_rank].release()
                return False

        stage_path = os.path.join(self._get_stage_path(), str(step))
        os.makedirs(stage_path, exist_ok=True)

        step_done_path = os.path.join(
            self._get_stage_path(), str(step) + ".done"
        )
        os.makedirs(step_done_path, exist_ok=True)

        step_done_file = os.path.join(step_done_path, str(self._node_rank))

        write_success = False
        if os.path.exists(step_done_file):
            logger.info(f"Rank {self._node_rank} already done for step {step}")
            write_success = True
        else:
            # save to stage path for each local rank
            futures = []
            for i in range(self.num_shard):
                future = self._executor.submit(_save_stage, i, stage_path)
                futures.append(future)

            success_count = 0
            for (i, future) in enumerate(futures):
                if future.result():
                    success_count += 1
                else:
                    logger.error(
                        f"Rank {i} save checkpoint failed for step {step}"
                    )

            if success_count == self.num_shard:
                # all local rank done success
                with open(step_done_file, "w") as f:
                    f.write("done")
                write_success = True

        if not write_success:
            logger.error(
                f"Rank {self._node_rank} save checkpoint failed for "
                f"step {step}"
            )
            return

        # commit checkpoint
        if self._is_agent_rank_0:
            self._commit_checkpoint(
                step,
                step_done_dir=step_done_path,
                tmp_path=stage_path,
                target_path=ckpt_path,
            )

        self._writing_storage = False

    def _commit_checkpoint(
        self, step, step_done_dir, tmp_path, target_path, timeout=60
    ):
        """
        Commit checkpoint from stage dir to target dir.

        This method is called by agent rank 0, it will check if all agent rank
        write finish, if true, it will commit checkpoint from stage dir to
        target dir.
        """
        logger.info(
            f"Start commit checkpoint tmp_path: {tmp_path}, "
            f"path: {target_path}"
        )
        start_time = time.time()
        while True:

            # check all local rank done
            if len(os.listdir(step_done_dir)) == self._node_num:
                # all local rank done
                logger.info(f"All agent done for step {tmp_path}")

                # commit checkpoint
                shutil.move(tmp_path, target_path)

                self.update_tracker_file(step)

                # clean stage dir
                shutil.rmtree(step_done_dir)
                logger.info(
                    f"Commit checkpoint tmp_path: {tmp_path}, "
                    f"path: {target_path}"
                )
                break

            # timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error(
                    f"Commit checkpoint timeout, tmp_path: {tmp_path},"
                    f"path: {target_path}, elapsed_time: {elapsed_time}"
                )
                # clean stage dir
                shutil.rmtree(tmp_path)
                shutil.rmtree(step_done_dir)
                break

            time.sleep(2)

    def _acquire_all_locks(self):
        """
        Acquire all locks of the shared memory, if not all locks are acquired,
        we should release acquired locks.
        """
        acquired = []
        for lock in self._shm_lock:
            acquired.append(lock.acquire(blocking=False))

        if not all(acquired):
            for i, lock in enumerate(self._shm_lock):
                if acquired[i]:
                    lock.release()
            return False

        return True

    def save_shm_to_storage(self):
        """
        Save the state dict in the shared memory into the storage. The agent
        can call the method to save the state dict into the storage if the
        training process fails or the agent wants to restart training
        processes.
        """
        logger.info("Agent terminated, save latest checkpoint to storage")
        if any([not shm for shm in self._tensor_shm]):
            return

        if self._writing_storage:
            logger.info("Saver is writing to storage, waiting...")
            start = time.time()
            while self._writing_storage:
                time.sleep(10)
                elapsed_time = time.time() - start
                if elapsed_time > 120:
                    logger.error("Saver writing to storage, timeout")
                    return

        if not self._acquire_all_locks():
            # The training process does not release the lock because it fails
            # when writing the state dict into the shared memory. The shared
            # memory may be dirty and the saver cannot save it to the storage.
            return

        meta_dict = self._shared_ckpt_meta[0].get()
        step = meta_dict["step"]
        # we need to release the locks before save to storage

        for lock in self._shm_lock:
            lock.release()
        self._save_shm_to_storage(step)

        for lock in self._shm_lock:
            lock.release()

        logger.info(
            "Save the checkpointing state dict from the shared "
            f"memory to storage, step: {step}."
        )


class CheckpointEngine(metaclass=ABCMeta):
    def __del__(self):
        self.close()

    @abstractmethod
    def save_to_memory(self, state_dict, step):
        """
        Save the state dict into the memory

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
        """
        pass

    @abstractmethod
    def save_to_storage(self, state_dict, path, step):
        """
        Save the state_dict into the path of storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
            step (int): the iteration step.
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

    @abstractmethod
    def close(self):
        """Close the resource."""
        pass


class NoShardingCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict into
    the shared memory and notify the agent in main process to
    asynchronously save the state dict from the shared memory into
    the storage. Writing to memory is significantly quicker
    than writing to storage. The engine only blocks the training
    with a little time. Users can frequently call `save_to_memory` in
    the training loop and call `save_to_storage`.

    If the training process fail, the agent in main process can continuely
    saves the the state dict from the shared memory into the storage.
    The engine saves the model and optimizer state dict without sharding
    in a local or DDP job.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.

    Examples::
        >>> engine = NoShardingCheckpointEngine(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>> )
        >>> for step, data in enumerate(dataloader):
        >>>     ...
        >>>     state_dict = model.state_dict()
        >>>     if step % 5 == 0:
        >>>         engine.save_to_memory(state_dict, step)
        >>>     elif step % 100 == 0:
        >>>         path = f"/tmp/checkpoint/ckpt-{step}.pt"
        >>>         engine.save_to_storage(state_dict, path, step)
        >>> sate_dict = engine.load()
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ["LOCAL_RANK"])
            saver_ranks = self._get_saver_ranks()
            self._saver_group = dist.new_group(
                ranks=saver_ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )
        else:
            self._rank = 0
            self._local_rank = int(os.getenv("LOCAL_RANK", 0))
            self._saver_group = None

        self._buffer_size = 0
        self._cached_step = 0
        self._restart_count = env_utils.get_torch_restart_count()

        meta_name = _CKPT_META_NAME_PREFIX + str(0)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=False)
        lock_name = _SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=False)
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=False)
        self._shm_name = _TENSOR_SHM_NAME_PREFIX + str(0)
        self._tensor_shm = None
        self._notify_agent_to_create_saver()

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
        return save_ranks

    def _notify_agent_to_create_saver(self):
        if self._local_rank != 0:
            return
        if self._restart_count > 0:
            # Only local rank 0 notify to initialize the saver in
            # the main process at the first start.
            # Avoid the lock is locked by a failed process.
            self._shm_lock.release()
            return
        queue = SharedQueue(name="factory")
        num_shard = env_utils.get_local_world_size()
        class_meta = SaverClassMeta(
            module_path="dlrover.python.elastic_agent.torch.ckpt_saver",
            class_name="NoShardingSaver",
            init_args={
                "checkpoint_dir": self.checkpoint_dir,
                "num_shard": num_shard,
            },
        )
        queue.put(class_meta)
        queue.close()

    def __del__(self):
        self.close()

    def close(self):
        if self._tensor_shm:
            self._tensor_shm.close()

    @timer
    def save_to_memory(self, state_dict, step):
        """
        Synchonously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
        """
        if self._local_rank != 0:
            return
        if "step" not in state_dict:
            state_dict["step"] = step
        if _WIRTING_SHM in state_dict:
            raise ValueError(f"state_dict cannot have the key {_WIRTING_SHM}.")

        if self._tensor_shm is None:
            self._make_state_dict_buffer(state_dict)
        acquired = self._shm_lock.acquire(blocking=False)
        all_rank_ready = _check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_lock.release()
            return
        self._copy_state_dict_to_shm(state_dict)

        if acquired:
            self._shm_lock.release()
        self._cached_step = step

    def _create_tensor_meta(self, value: torch.Tensor):
        """
        Create a tensor meta of a tensor and compute the total
        size of the state dict.
        """
        if not torch.is_tensor(value):
            return value
        meta = TensorMeta(
            shape=tuple(value.shape),  # type: ignore
            dtype=value.dtype,
            element_size=value.element_size(),
            numel=value.numel(),
            offset=self._buffer_size,
        )
        self._buffer_size += value.numel() * value.element_size()
        return meta

    def _make_state_dict_buffer(self, state_dict):
        """
        Make the shared memory to store the state dict.
        """
        meta_dict = _traverse_state_dict(state_dict, self._create_tensor_meta)

        self._shared_ckpt_meta.update(meta_dict)
        self._tensor_shm = _create_shared_memory(
            name=self._shm_name,
            create=True,
            size=self._buffer_size,
        )

    def _copy_state_dict_to_shm(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """
        meta_dict = self._shared_ckpt_meta.get(local=True)
        meta_dict[_WIRTING_SHM] = True
        self._shared_ckpt_meta.update(meta_dict)
        _tarverse_copy_to_shm(state_dict, meta_dict, self._tensor_shm.buf)
        meta_dict[_WIRTING_SHM] = False
        self._shared_ckpt_meta.update(meta_dict)

    @timer
    def save_to_storage(self, state_dict, path, step):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
            step (int): the iteration step.
        """
        if self._rank != 0:
            return
        if step > self._cached_step:
            self.save_to_memory(state_dict, step)
        if path:
            self._to_save_queue.put(path)

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self._load_from_shared_memory()
        if state_dict:
            return state_dict
        state_dict = self._load_from_storage(resume_path)
        return state_dict

    def _load_from_shared_memory(self):
        """
        Load the state dict from the shared memory.

        Returns:
            A dict.
        """
        meta_dict = self._shared_ckpt_meta.get()
        if not meta_dict or meta_dict.get(_WIRTING_SHM, False):
            return None
        if self._tensor_shm is None:
            self._tensor_shm = _create_shared_memory(
                self._shm_name,
                create=False,
            )
        if not self._tensor_shm:
            return None
        state_dict = _read_state_dict_from_shm(meta_dict, self._tensor_shm)
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
        if resume_path:
            state_dict = torch.load(resume_path)
        else:
            state_dict = _load_from_historic_checkpoint(self.checkpoint_dir)
        return state_dict


class ShardingCheckpointEngine(CheckpointEngine, ABC):
    """
    The engine to save the sharding model and optimizer state dict
    into the memory and storage. We can use it to save the model and optimizer
    using FSDP, Zero-3 or Megatron-LM.
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._saver_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=30),
            )

        else:
            self._rank = 0
            self._local_rank = int(os.getenv("LOCAL_RANK", 0))
            self._saver_group = None

        self._buffer_size = 0
        self._cached_step = 0
        self._restart_count = env_utils.get_torch_restart_count()

        meta_name = _CKPT_META_NAME_PREFIX + str(self._local_rank)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=False)
        lock_name = _SHM_LOCK_NAME_PREFIX + str(self._local_rank)
        self._shm_lock = SharedLock(name=lock_name, create=False)

        if self._rank == 0:
            self._to_save_queue = SharedQueue(
                name=_SAVE_STEP_QNAME_PREFIX + str(0), create=False
            )
        else:
            self._to_save_queue = None

        self._shm_name = _TENSOR_SHM_NAME_PREFIX + str(self._local_rank)
        self._tensor_shm = None
        self._notify_agent_to_create_saver()

    def _notify_agent_to_create_saver(self):
        if self._rank == 0:

            if self._restart_count > 0:
                logger.info(
                    f"Restart count is {self._restart_count}, release lock"
                )
                self._shm_lock.release()
                return

            queue = SharedQueue(name="factory")
            num_shard = env_utils.get_local_world_size()

            # get class module_path
            clazz = self.get_saver_class()
            module_path = clazz.__module__
            class_name = clazz.__name__

            class_meta = SaverClassMeta(
                module_path=module_path,
                class_name=class_name,
                init_args={
                    "checkpoint_dir": self.checkpoint_dir,
                    "num_shard": num_shard,
                },
            )
            queue.put(class_meta)
            queue.close()

    @abstractmethod
    def get_saver_class(self):
        pass

    @timer
    def save_to_memory(self, state_dict, step):

        if "step" not in state_dict:
            state_dict["step"] = step

        if _WIRTING_SHM in state_dict:
            raise ValueError(f"state_dict cannot have the key {_WIRTING_SHM}.")

        if self._tensor_shm is None:
            self._make_state_dict_buffer(state_dict)

        acquired = self._shm_lock.acquire(blocking=False)
        all_rank_ready = _check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_lock.release()
            return
        self._copy_state_dict_to_shm(state_dict)

        if acquired:
            self._shm_lock.release()
        self._cached_step = step

    @timer
    def _copy_state_dict_to_shm(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """
        meta_dict = self._shared_ckpt_meta.get(local=True)
        meta_dict[_WIRTING_SHM] = True
        self._shared_ckpt_meta.update(meta_dict)
        _tarverse_copy_to_shm(state_dict, meta_dict, self._tensor_shm.buf)
        meta_dict[_WIRTING_SHM] = False
        self._shared_ckpt_meta.update(meta_dict)

    def _create_tensor_meta(self, value: torch.Tensor):
        """
        Create a tensor meta of a tensor and compute the total
        size of the state dict.
        """
        if not torch.is_tensor(value):
            return value
        meta = TensorMeta(
            shape=tuple(value.shape),  # type: ignore
            dtype=value.dtype,
            element_size=value.element_size(),
            numel=value.numel(),
            offset=self._buffer_size,
        )
        self._buffer_size += value.numel() * value.element_size()
        return meta

    def _make_state_dict_buffer(self, state_dict):
        """
        Make the shared memory to store the state dict.
        """
        meta_dict = _traverse_state_dict(state_dict, self._create_tensor_meta)

        self._shared_ckpt_meta.update(meta_dict)
        self._tensor_shm = _create_shared_memory(
            name=self._shm_name,
            create=True,
            size=self._buffer_size,
        )

    def close(self):
        if self._tensor_shm:
            self._tensor_shm.close()

    def _init_shared_objs(self):
        meta_name = _CKPT_META_NAME_PREFIX + str(self._local_rank)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=False)
        lock_name = _SHM_LOCK_NAME_PREFIX + str(self._local_rank)
        self._shm_lock = SharedLock(name=lock_name, create=False)
        self._shm_name = _TENSOR_SHM_NAME_PREFIX + str(self._local_rank)

        # only agent rank 0 notify saver to save
        if self._local_rank == 0:
            qname = _SAVE_STEP_QNAME_PREFIX + str(self._local_rank)
            self._to_save_queue = SharedQueue(name=qname, create=False)

    def save_to_storage(self, state_dict, path, step):
        if step > self._cached_step:
            self.save_to_memory(state_dict, step)

        if self._local_rank == 0:
            self._to_save_queue.put(step)
