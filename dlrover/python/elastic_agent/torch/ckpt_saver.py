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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Mapping, Tuple

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


class SharedMemoryHandler(object):
    """
    The handler to write and read the shared memory with the state dict
    of PyTorch Module.

    Args:
        local_rank (int): the local rank of the process on a node.
        on_host (bool): the handler is on the host if True, otherwise,
            the handler is on the device.
    """

    def __init__(self, local_rank, on_host=True):
        self._buffer_size = 0
        meta_name = _CKPT_META_NAME_PREFIX + str(local_rank)
        self._tensor_meta = SharedDict(name=meta_name, create=on_host)
        self._shm_name = _TENSOR_SHM_NAME_PREFIX + str(local_rank)
        self._tensor_shm = None

    def close(self):
        if self._tensor_shm:
            self._tensor_shm.close()

    def unlink(self):
        if self._tensor_shm:
            self._tensor_shm.unlink()
        if self._tensor_meta:
            self._tensor_meta.unlink()

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

    def make_state_dict_buffer(self, state_dict):
        """
        Make the shared memory to store the state dict.
        """
        if self._tensor_shm is not None:
            return

        meta_dict = _traverse_state_dict(state_dict, self._create_tensor_meta)
        self._tensor_meta.update(meta_dict)
        self._tensor_shm = _create_shared_memory(
            name=self._shm_name,
            create=True,
            size=self._buffer_size,
        )

    def save_state_dict(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """
        meta_dict = self._tensor_meta.get(local=True)
        meta_dict[_WIRTING_SHM] = True
        self._tensor_meta.update(meta_dict)
        _tarverse_copy_to_shm(state_dict, meta_dict, self._tensor_shm.buf)
        meta_dict[_WIRTING_SHM] = False
        self._tensor_meta.update(meta_dict)

        meta_dict = self._tensor_meta.get(local=True)

    def load_state_dict(self):
        """
        Load the state dict from the shared memory.

        Returns:
            A dict.
        """
        meta_dict = self._tensor_meta.get()
        step = meta_dict.get("step", 0)
        if not meta_dict or meta_dict.get(_WIRTING_SHM, False):
            return step, {}
        if self._tensor_shm is None:
            self._tensor_shm = _create_shared_memory(
                self._shm_name,
                create=False,
            )
        if not self._tensor_shm:
            return step, {}

        state_dict = _read_state_dict_from_shm(meta_dict, self._tensor_shm)
        return step, state_dict

    def empty(self):
        return self._tensor_shm is None


class CheckpointSaver(metaclass=ABCMeta):
    """
    CheckpointSaver saves the state dict from the shared memory into
    the storage.

    Attributes:
        checkpoint_dir (str): the directory to save the checkpointing state
            dict to the storage if the training process fails.
    """

    _saver_instance = None

    def __init__(
        self,
        checkpoint_dir,
    ):
        self.checkpoint_dir = checkpoint_dir

    @classmethod
    def start_async_saving_ckpt(cls):
        """
        Start a thread to asynchronously save the checkpoint state dict
        from the shared memory into the storage. Firstly, it waits that
        the training process notify the saver class to create a saver.

        Args:
            num_proc: the number of training process, i.e. local world size.
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


class NoShardingSaver(CheckpointSaver):
    """
    The saver only saves the state dict without sharding
    from the shared memory created by local rank 0 to the storage.
    """

    def __init__(self, checkpoint_dir) -> None:
        super().__init__(checkpoint_dir)
        # Only local rank 0 save the state dict to memory in DDP.
        self._shm_handler = SharedMemoryHandler(0)
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=True)
        lock_name = _SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=True)

    def __del__(self):
        self.close()

    def close(self):
        if self._shm_handler:
            self._shm_handler.close()
            self._shm_handler.unlink()
        self._to_save_queue.unlink()
        self._shm_lock.unlink()

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Async checkpoint saver starts!")
        while True:
            path = self._to_save_queue.get()
            self._shm_lock.acquire()
            logger.info(
                "Save checkpoint from the shared memory "
                f"into the storage {path}."
            )
            _, state_dict = self._shm_handler.load_state_dict()
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
        if self._shm_handler.empty():
            return
        acquired = self._shm_lock.acquire()
        if not acquired:
            # The training process does not release the lock because it fails
            # when writing the state dict into the shared memory. The shared
            # memory may be dirty and the saver cannot save it to the storage.
            return
        step, state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint-{step}/checkpoint.pt"
            )
            self._persist_to_storage(state_dict, path)
            logger.info(
                "Save the checkpointing state dict from the shared "
                f"memory to {path}."
            )
        self._shm_lock.release()


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

        lock_name = _SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=False)
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=False)
        self._shm_handler = SharedMemoryHandler(0, on_host=False)
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
        class_meta = SaverClassMeta(
            module_path="dlrover.python.elastic_agent.torch.ckpt_saver",
            class_name="NoShardingSaver",
            init_args={"checkpoint_dir": self.checkpoint_dir},
        )
        queue.put(class_meta)
        queue.unlink()

    def __del__(self):
        self.close()

    def close(self):
        self._shm_handler.close()

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

        self._shm_handler.make_state_dict_buffer(state_dict)
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
        self._shm_handler.save_state_dict(state_dict)

        if acquired:
            self._shm_lock.release()
        self._cached_step = step

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
        step, state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            return step, state_dict
        state_dict = self._load_from_storage(resume_path)
        step = state_dict.get("step", 0)
        return step, state_dict

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
