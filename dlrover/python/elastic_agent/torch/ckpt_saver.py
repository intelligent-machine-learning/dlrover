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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Mapping, Tuple

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import (
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
)

_SAVE_STEP_QNAME_PREFIX = "checkpoint_lock_rank_"
_CKPT_META_NAME_PREFIX = "checkpoint_meta_shard_"
_TENSOR_SHM_NAME_PREFIX = "checkpoint_shm_shard_"
_SHM_LOCK_NAME_PREFIX = "shm_shard_"
_DLROVER_CKPT_KEY = "_DLORVER_CKPT_CONFIG"

_SAVE_EVENT_NAME = "SAVE_CHECKPOINT"
_UPDATE_EVENT_NAME = "UPDATE_SHARD_NUM"


@dataclass
class SaverClassMeta:
    module_path: str = ""
    class_name: str = ""
    init_args: Dict[str, str] = None  # type: ignore


@dataclass
class TensorMeta:
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


@dataclass
class CheckpointShardConfig:
    """
    The configuration of a checkpointing shard on the training process.

    Attrbiutes:
        step (int): the global interation step.
        ckpt_name (str): the path to save the checkpoint shard.
        writing_shm (bool): the flag whether the training process is writing
            the state dict into the shared memory.
    """

    step: int = 0
    ckpt_name: str = ""
    writing_shm: bool = False


@dataclass
class SaveEvent:
    name: str = ""
    step: int = 0
    global_shard_num: int = 0


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(f"Function {func.__name__} cost {t}s")
        return result

    return wrapper


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
        if shm.size != size:
            logger.info("Re-create a new memory buffer.")
            shm.unlink()
            shm = SharedMemory(
                name=name,
                create=create,
                size=size,
            )
    return shm


def _check_all_rank_ready(group, ready):
    """
    Check weather all ranks are ready.
    """
    if not group:
        return ready
    value = 0 if ready else 1
    t = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(t, group=group)
    return t == 0


def _traverse_copy_to_shm(value, meta, buffer):
    if isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(v, (Mapping, List)):
                m = meta[k]
                _traverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                m = meta[k]
                _write_shared_memory(v, m, buffer)
            else:
                meta[k] = v
    elif isinstance(value, List):
        for i, v in enumerate(value):
            if isinstance(v, (Mapping, List)):
                m = meta[i]
                _traverse_copy_to_shm(v, m, buffer)
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


class SharedMemoryHandler(object):
    """
    The handler to write and read the shared memory with the state dict
    of PyTorch Module.

    Args:
        local_rank (int): the local rank of the process on a node.
        host (bool): the handler is on the host if True, otherwise,
            the handler is on the device.
    """

    def __init__(self, local_rank, host=True):
        self._buffer_size = 0
        meta_name = _CKPT_META_NAME_PREFIX + str(local_rank)
        self._tensor_meta = SharedDict(name=meta_name, create=host)
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

    def reset(self):
        self.close()
        if self._tensor_shm:
            self._tensor_shm.unlink()
        self._tensor_meta.set({})
        self._tensor_shm = None

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

    def save_state_dict(self, step, state_dict, ckpt_name=""):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """
        if self._tensor_shm is None:
            meta_dict = _traverse_state_dict(
                state_dict, self._create_tensor_meta
            )
            self.init_tensor_shm(create=True, size=self._buffer_size)
        else:
            meta_dict = self._tensor_meta.get(local=True)

        conf = CheckpointShardConfig(
            step=step,
            ckpt_name=ckpt_name,
            writing_shm=True,
        )
        meta_dict[_DLROVER_CKPT_KEY] = conf
        self._tensor_meta.set(meta_dict)
        _traverse_copy_to_shm(state_dict, meta_dict, self._tensor_shm.buf)
        conf.writing_shm = False
        self._tensor_meta.set(meta_dict)

    def load_state_dict(self):
        """
        Load the state dict from the shared memory.

        Returns:
            Tuple(int, dict): The first value is the iteration step,
                the second value is the state dict.
        """
        meta_dict = self._tensor_meta.get()
        default_config = CheckpointShardConfig()
        config = meta_dict.get(_DLROVER_CKPT_KEY, default_config)
        if not meta_dict or config.writing_shm:
            return {}
        if self._tensor_shm is None:
            self.init_tensor_shm(create=False)
        if not self._tensor_shm:
            return {}

        state_dict = _read_state_dict_from_shm(meta_dict, self._tensor_shm)
        state_dict.pop(_DLROVER_CKPT_KEY, None)
        return state_dict

    def empty(self):
        meta_dict = self._tensor_meta.get()
        config: CheckpointShardConfig = meta_dict.get(_DLROVER_CKPT_KEY, None)
        if config is None or config.step == 0:
            return True
        return False

    def init_tensor_shm(self, create=False, size=0):
        self._tensor_shm = _create_shared_memory(
            self._shm_name, create=create, size=size
        )

    def get_checkpoint_config(self):
        """
        Get the configuration of checkpointing state dict in the shared
        memory.

        Returns:
            A CheckpointShardConfig instance.
        """
        meta_dict = self._tensor_meta.get()
        default_config = CheckpointShardConfig()
        config = meta_dict.get(_DLROVER_CKPT_KEY, default_config)
        return config


class CheckpointSaver(metaclass=ABCMeta):
    """
    CheckpointSaver saves the state dict from the shared memory into
    the storage.

    Attributes:
        checkpoint_dir (str): the directory to save the checkpointing state
            dict to the storage if the training process fails.
        local_shard_num (int): the number of model/optimizer shards
            on the node.
        global_shard_num (int): the number of model/optimizer shards
            across all nodes.
    """

    _saver_instance = None
    _STAGE_DIR = "._dlrover_ckpt_stage"

    def __init__(
        self, checkpoint_dir, local_shard_num=1, global_shard_num=1
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.local_shard_num = local_shard_num
        self.global_shard_num = global_shard_num
        self._node_rank = env_utils.get_node_rank()
        self._is_agent_rank_0 = self._node_rank == 0
        self._shm_handlers: List[SharedMemoryHandler] = []
        self._shm_locks: List[SharedLock] = []

        # Indicate whether the saver is writing state to storage
        self._writing_storage = False
        # The latest step to save the checkpoint.
        self._latest_step = 0
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._event_queue = SharedQueue(name=qname, create=True)
        for i in range(self.local_shard_num):
            self._shm_handlers.append(SharedMemoryHandler(i))
            lock_name = _SHM_LOCK_NAME_PREFIX + str(i)
            self._shm_locks.append(SharedLock(name=lock_name, create=True))
        self._executor = ThreadPoolExecutor(
            max_workers=self.local_shard_num, thread_name_prefix="ckpt_saver-"
        )

    def __del__(self):
        self.close()

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

    def close(self):
        """Clear the resource of the shared objects."""
        for i in range(self.local_shard_num):
            if self._shm_handlers[i]:
                self._shm_handlers[i].close()
                self._shm_handlers[i].unlink()
            self._shm_locks[i].unlink()
        self._event_queue.unlink()
        self._executor.shutdown()

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Async checkpoint saver starts!")
        while True:
            event: SaveEvent = self._event_queue.get()
            if (
                event.name == _UPDATE_EVENT_NAME
                and event.global_shard_num != self.global_shard_num
            ):
                logger.info(
                    "Reset the shard memory because the number of "
                    "global shards changes"
                )
                self._reset_shared_memory()
            elif event.name == _SAVE_EVENT_NAME:
                logger.info(
                    f"ShardingSaver save checkpoint to storage, event {event}"
                )
                self.save_step_checkpoint(event.step)

    def _reset_shared_memory(self):
        for shm_handler in self._shm_handlers:
            shm_handler.reset()

    def _save_shard(
        self, step, local_shard_id: int, ckpt_path: str, step_done_dir: str
    ):
        """Save the shard of state dict into the storage."""
        try:
            shm_handler = self._shm_handlers[local_shard_id]
            shm_lock = self._shm_locks[local_shard_id]
            if shm_handler.empty():
                shm_handler.init_tensor_shm(create=False)

            shm_lock.acquire()
            config = shm_handler.get_checkpoint_config()
            if config.step != step:
                shm_lock.release()
                logger.error(
                    f"The step {step} in event is no equal "
                    f"to step {config.step} in memory."
                )
                return

            logger.info(
                f"Local rank {local_shard_id} Save checkpoint from the shared "
                f"memory into the storage {ckpt_path}."
            )
            state_dict = shm_handler.load_state_dict()
            state_dict.pop(_DLROVER_CKPT_KEY, None)
            self.persist_to_storage(state_dict, ckpt_path)
            shm_lock.release()
            global_shard_id = (
                self.local_shard_num * self._node_rank + local_shard_id
            )
            step_done_file = os.path.join(step_done_dir, str(global_shard_id))
            with open(step_done_file, "w") as f:
                f.write("done")
            return True

        except Exception as e:
            logger.error(
                f"Rank {local_shard_id} save checkpoint failed, error: {e}",
                exc_info=True,
            )
            shm_lock.release()
            return False

    def _any_rank_locked(self):
        """Verify that the shared memory of any rank is locked."""
        all_locked = []
        for lock in self._shm_locks:
            all_locked.append(lock.locked())
        return any(all_locked)

    def _get_checkpoint_done_dir(self, step):
        """Get the directory of the done files."""
        done_dir = os.path.join(
            self.checkpoint_dir, self._STAGE_DIR, str(step) + ".done"
        )
        os.makedirs(done_dir, exist_ok=True)
        return done_dir

    def save_shm_to_storage(self, timeout=120):
        """
        Save the state dict in the shared memory into the storage. The agent
        can call the method to save the state dict into the storage if the
        training process fails or the agent wants to restart training
        processes.
        """
        if any([handler.empty() for handler in self._shm_handlers]):
            logger.info(
                "Skip because no any memory buffer with the state dict."
            )
            return

        if self._writing_storage:
            logger.info("Saver is writing to storage, waiting...")
            start = time.time()
            while self._writing_storage:
                time.sleep(2)
                elapsed_time = time.time() - start
                if elapsed_time > timeout:
                    logger.error("Saver writing to storage, timeout")
                    return
        if self._any_rank_locked():
            # The training process does not release the lock because it fails
            # when writing the state dict into the shared memory. The shared
            # memory may be dirty and the saver cannot save it to the storage.
            return
        steps = []
        for shm_handler in self._shm_handlers:
            config = shm_handler.get_checkpoint_config()
            steps.append(config.step)
        if len(set(steps)) > 1:
            logger.error(
                "Skip because steps in shards are not "
                f"inconsistent: {steps}"
            )
            return
        step = steps[0]
        if step > self._latest_step:
            self.save_step_checkpoint(step)
            logger.info(
                "Save the checkpointing state dict from the shared "
                f"memory to storage, step: {steps}."
            )

    @abstractmethod
    def save_step_checkpoint(self, step: int):
        """
        Save the checkpoint of a step into the storage.

        Args:
            step (int): the iteration step.
        """
        pass

    @abstractmethod
    def persist_to_storage(self, state_dict, ckpt_path):
        """
        Persist the state dict to a storage path.

        Args:
            state_dict (dict): the state dict of PyTorch modules.
            ckpt_path (str): the path of storaget to persist the state dict.
        """
        pass

    @abstractmethod
    def commit_checkpoint(self, step: int, step_done_dir: str, timeout=600):
        """
        Commit a checkpoint if all shards are saved to the storage.

        Args:
            step (int): the iteration step.
            step_done_dir (str): the directory to save the
                done file of each shard.
            timeout (int): the timeout to wait that all shards are saved,
                default 600s.
        """
        pass

    @abstractmethod
    def update_tracker_file(self, step: int):
        """
        Update the checkpoint tracker file on the storage after the
        checkpoing state dict is saved to the storage.

        Args:
            step (int): the iteration step.
        """
        pass


class AsyncCheckpointSaver(CheckpointSaver):
    """
    The saver asynchronously save the checkpointing state dict from
    the shared memory to the storage. It launches threads to save
    each shard in the shared memory. The thread saves each shard
    to a unique path and writes a done file after the saving is
    finshied. The path is generated by the training framework like
    Megatron-LM, DeepSpeed or users' definition. The saver updates
    the tracer file with the step if the number of done file is equal
    to the number of shard.
    """

    @classmethod
    def get_checkpoint_tracker_filename(cls, checkpoint_dir):
        """
        Get the path of tracker file to record the latest checkpointing
        step.

        Args:
            checkpoint_dir (str): the checkpoint directory.

        Returns:
            str: the path of tracker file.
        """
        fname = CheckpointConstant.TRACER_FILE_NAME
        return os.path.join(checkpoint_dir, fname)

    def update_tracker_file(self, step):
        """
        Write the step into the tracker file.

        Args:
            step (int): the checkpointing step.
        """
        tracker_filename = self.get_checkpoint_tracker_filename(
            self.checkpoint_dir
        )
        with open(tracker_filename, "w") as f:
            f.write(str(step))

    def save_step_checkpoint(self, step: int):
        """
        Save the checkpoint of a step into the storage.

        Args:
            step (int): the iteration step.
        """
        self._writing_storage = True

        step_done_dir = self._get_checkpoint_done_dir(step)
        os.makedirs(step_done_dir, exist_ok=True)

        write_success = False
        # save to stage path for each local rank
        futures: List[Future] = []
        for i in range(self.local_shard_num):
            ckpt_config = self._shm_handlers[i].get_checkpoint_config()
            future: Future = self._executor.submit(
                self._save_shard,
                step,
                i,
                ckpt_config.ckpt_name,
                step_done_dir,
            )
            futures.append(future)

        success_count = 0
        for (i, future) in enumerate(futures):
            if future.result():
                success_count += 1
            else:
                logger.error(
                    f"Fail to save checkpoint shared {i} for step {step}"
                )

        if success_count == self.local_shard_num:
            write_success = True
            self._latest_step = step

        if not write_success:
            logger.error(
                f"Rank {self._node_rank} save checkpoint failed for "
                f"step {step}"
            )
            return

        # commit checkpoint
        if self._is_agent_rank_0:
            self.commit_checkpoint(step, step_done_dir)

        self._writing_storage = False

    def persist_to_storage(self, state_dict, path):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        checkpoint_dir = os.path.dirname(path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state_dict, path)

    def commit_checkpoint(self, step: int, step_done_dir: str, timeout=600):
        """
        The node rank 0 will update the tracker file with the step
        after the number of done files is equal to the number of shard.

        Args:
            step (int): the iteration step.
            step_done_dir (str): the directory to save the
                done file of each shard.
            timeout (int): the timeout to wait that all shards are saved,
                default 600s.
        """
        start_time = time.time()
        while True:
            if len(os.listdir(step_done_dir)) == self.global_shard_num:
                # all local rank done
                self.update_tracker_file(step)

                # clean stage dir
                shutil.rmtree(step_done_dir)
                logger.info(
                    f"All agents finish saving checkpoint for step {step}"
                )
                break
            # timeout
            elapsed_time = round(time.time() - start_time, 2)
            if elapsed_time > timeout:
                logger.error(
                    f"Commit checkpoint timeout for step {step}, "
                    f"elapsed_time: {elapsed_time}"
                )
                # clean stage dir
                shutil.rmtree(step_done_dir)
                break

            time.sleep(2)


class AtorchFSDPShardingSaver(CheckpointSaver):
    """
    This saver saves the FSDP sharding state dict across all ranks.
    """

    def save_step_checkpoint(self, step):
        """
        Save the checkpoint of a step into the storage.

        Args:
            step (int): the iteration step.
        """
        logger.info(
            f"Rank {self._node_rank} start save checkpoint to storage, "
            f"step: {step}"
        )
        self._writing_storage = True
        ckpt_path = self.get_ckpt_path(step)

        if os.path.exists(ckpt_path):
            logger.info(f"Checkpoint for step {step} already exists, skip")
            self._writing_storage = False
            return

        stage_path = self._get_tmp_ckpt_dir(step)
        step_done_dir = self._get_checkpoint_done_dir(step)

        write_success = False
        # save to stage path for each local rank
        futures: List[Future] = []
        for i in range(self.local_shard_num):
            future = self._executor.submit(
                self._save_shard, step, i, stage_path, step_done_dir
            )
            futures.append(future)

        success_count = 0
        for (i, future) in enumerate(futures):
            if future.result():
                success_count += 1
            else:
                logger.error(
                    f"Rank {i} save checkpoint failed for step {step}"
                )

        if success_count == self.local_shard_num:
            write_success = True

        if not write_success:
            logger.error(
                f"Rank {self._node_rank} save checkpoint failed for "
                f"step {step}"
            )
            return

        # commit checkpoint
        if self._is_agent_rank_0:
            self.commit_checkpoint(
                step,
                step_done_dir=step_done_dir,
                tmp_path=stage_path,
                target_path=ckpt_path,
            )

        self._writing_storage = False

    def get_ckpt_path(self, step: int):
        """
        User can override the method to define the checkpoint name.

        Args:
            step (int): the iteration step.
        """
        name = f"{CheckpointConstant.CKPT_NAME_PREFIX}{step}"
        return os.path.join(self.checkpoint_dir, name)

    def _get_tmp_ckpt_dir(self, step: int):
        """
        Get the temp directory to save the latest checkpoint. After all
        shards are saved, the saver will move the directory to a
        regular directory defined by users.
        """
        ckpt_name = os.path.join(
            self.checkpoint_dir, self._STAGE_DIR, str(step)
        )
        os.makedirs(ckpt_name, exist_ok=True)
        return ckpt_name

    def commit_checkpoint(  # type: ignore
        self,
        step: int,
        step_done_dir: str,
        tmp_path: str,
        target_path: str,
        timeout=600,
    ):
        """
        Commit checkpoint from stage dir to target dir.

        This method is called by agent rank 0, it will check if all agent rank
        write finish, if true, it will commit checkpoint by move stage dir to
        target dir.

        Args:
            step (int): the iteration step.
            step_donr_dir (str): the directory to save the done file of
                each shard.
            tmp_path: the temp directory path to save the latest checkpoint.
            target_path: the regular diectory path to save the checkpoint.
            timeout (int): the timeout to wait all shards are saved,
                default 600s.
        """
        logger.info(
            f"Start commit checkpoint tmp_path: {tmp_path}, "
            f"path: {target_path}"
        )
        start_time = time.time()
        while True:

            # check all local rank done
            if len(os.listdir(step_done_dir)) == self.global_shard_num:
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


class CheckpointEngine(metaclass=ABCMeta):
    """
    The engine runs in the training process and is called by the
    training program. It synchronously saves the checkpointing
    state dict into the CPU memory buffer and notifies the checkpoint
    saver to save the checkpoint from CPU memory buffer to the storage.

    Args:
        checkpoint_dir (str): the directory to save checkpoint.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self._rank = 0
            self._local_rank = int(os.getenv("LOCAL_RANK", 0))
        self._saver_group = None
        self._buffer_size = 0
        self._cached_step = 0
        self._restart_count = env_utils.get_torch_restart_count()
        # queue for agent to save to storage, only rank 0
        if self._rank == 0:
            self._event_queue = SharedQueue(
                name=_SAVE_STEP_QNAME_PREFIX + str(0), create=False
            )
        else:
            self._event_queue = None  # type: ignore
        # lock for shared memory
        local_shard_num = self.get_local_shard_num()
        self.local_shard_id = self._local_rank % local_shard_num
        lock_name = _SHM_LOCK_NAME_PREFIX + str(self.local_shard_id)
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
            event: SaveEvent = SaveEvent(
                name=_UPDATE_EVENT_NAME,
                global_shard_num=global_shard_num,
            )
            self._event_queue.put(event)

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

        if _DLROVER_CKPT_KEY in state_dict:
            raise ValueError(
                f"The state_dict cannot have the key {_DLROVER_CKPT_KEY}."
            )

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
        self._shm_handler.save_state_dict(step, state_dict, path)

        if acquired:
            self._shm_lock.release()
        self._cached_step = step

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
        pass

    @abstractmethod
    def load(self, resume_path=""):
        """
        Load the checkpointing state dict from the resume path.

        Returns:
            A dict.
        """
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
        >>>     path = f"/tmp/checkpoint-{step}.pt"
        >>>     if step % 5 == 0:
        >>>         engine.save_to_memory(step, state_dict, path)
        >>>     elif step % 100 == 0:
        >>>         engine.save_to_storage(step, state_dict, path)
        >>> sate_dict = engine.load()
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir)
        if dist.is_initialized():
            saver_ranks = self._get_saver_ranks()
            self._saver_group = dist.new_group(
                ranks=saver_ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )

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

    def get_local_shard_num(self):
        return 1

    def get_global_shard_num(self):
        return 1

    def get_saver_class(self):
        return AsyncCheckpointSaver

    @timer
    def save_to_storage(self, step, state_dict, path):
        """
        Asynchronously saves the state dict into the storage. It synchronously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            ckpt_name (str): the storage path to save the state dict.
                Note, the ckpt_name is used to save the state dict to storage
                only if the training process fails.
            path (int): the iteration step.
        """
        if self._rank != 0:
            return
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, path)
        event = SaveEvent(name=_SAVE_EVENT_NAME, step=step)
        if self._local_rank == 0:
            self._event_queue.put(event)

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            logger.info("Load the state dict from the CPU memory buffer.")
            state_dict.pop(_DLROVER_CKPT_KEY, None)
            return state_dict
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
        if resume_path:
            state_dict = torch.load(resume_path, map_location="cpu")
            return state_dict
        else:
            func = AsyncCheckpointSaver.get_checkpoint_tracker_filename
            tracker_filename = func(self.checkpoint_dir)
            if not os.path.exists(tracker_filename):
                return {}
            with open(tracker_filename, "r") as f:
                metastring = f.read().strip()
            iteration = int(metastring)
            name = f"{CheckpointConstant.CKPT_NAME_PREFIX}{iteration}.pt"
            path = os.path.join(self.checkpoint_dir, name)
            logger.info(f"Load the state dict from {path}")
            state_dict = torch.load(path, map_location="cpu")
            return state_dict


class FSDPShardingCheckpointEngine(CheckpointEngine):
    """
    The engine to save the sharding model and optimizer state dict
    shared across all ranks into the memory and storage.
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir)
        if dist.is_initialized():
            self._saver_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=30),
            )

    def save_to_storage(self, step, state_dict, path):
        if step > self._cached_step:
            self.save_to_memory(step, state_dict, path)

        save_event = SaveEvent(name=_SAVE_EVENT_NAME, step=step)
        if self._local_rank == 0:
            self._event_queue.put(save_event)


class MegatronCheckpointEngine(CheckpointEngine):
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
        >>> engine = MegatronCheckpointEngine(
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
            event = SaveEvent(name=_SAVE_EVENT_NAME, step=step)
            self._event_queue.put(event)

    def get_local_shard_num(self):
        local_world_size = env_utils.get_local_world_size()
        global_shard_num = self.get_global_shard_num()
        return min(local_world_size, global_shard_num)

    def get_global_shard_num(self):
        num = self._pp_world_size * self._tp_world_size
        return num

    def get_saver_class(self):
        return AsyncCheckpointSaver

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            return state_dict
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
        if resume_path:
            state_dict = torch.load(resume_path, map_location="cpu")
            return state_dict
        else:
            func = AsyncCheckpointSaver.get_checkpoint_tracker_filename
            tracker_filename = func(self.checkpoint_dir)
            if not os.path.exists(tracker_filename):
                return {}
            with open(tracker_filename, "r") as f:
                metastring = f.read().strip()
            iteration = int(metastring)
            ckpt_name = self._get_checkpoint_name(iteration)
            state_dict = torch.load(ckpt_name, map_location="cpu")
            return state_dict

    def _get_checkpoint_name(self, step):
        directory = "iter_{:07d}".format(step)
        # Use both the tensor and pipeline MP rank.
        if self._pp_world_size == 1:
            return os.path.join(
                self.checkpoint_dir,
                directory,
                "mp_rank_{:02d}".format(self._tp_rank),
                "model_optim_rng.pt",
            )
        return os.path.join(
            self.checkpoint_dir,
            directory,
            "mp_rank_{:02d}_{:03d}".format(self._tp_rank, self._pp_rank),
            "model_optim_rng.pt",
        )
