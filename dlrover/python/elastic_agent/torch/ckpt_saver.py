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
import json
import os
import pickle
import signal
import threading
import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

import dlrover.python.util.file_util as fu
from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    CheckpointConstant,
    NodeEnv,
    TrainingExceptionLevel,
)
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.multi_process import (
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
)
from dlrover.python.common.serialize import ClassMeta
from dlrover.python.elastic_agent.master_client import MasterClient

DLROVER_CKPT_CONFIG_KEY = "_DLORVER_CKPT_CONFIG"


class CheckpointSharedObjPrefix:
    SAVE_STEP_QNAME = "ckpt_lock_rank_"
    META_NAME = "ckpt_meta_"
    SHM_NAME = "ckpt_shm_"
    SHM_LOCK_NAME = "shm_lock_"


class CheckpointEventType(Enum):
    SAVE = auto()
    UPDATE_SHARD = auto()
    EXIT = auto()


@dataclass
class CheckpointEvent:
    type: CheckpointEventType = CheckpointEventType.SAVE
    step: int = 0
    global_shard_num: int = 0


@dataclass
class TensorMeta:
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


@dataclass
class CheckpointConfig:
    """
    The configuration of a checkpointing shard on the training process.

    Attributes:
        step (int): the global interation step.
        writing_shm (bool): the flag whether the training process is writing
            the state dict into the shared memory.
        paths (dict): the key is in ["model_state", "optim_state"] and the
            value is path.
    """

    rank: int = 0
    group_rank: int = 0
    world_size: int = 0
    step: int = 0
    writing_shm: bool = False
    paths: Dict[str, str] = None  # type: ignore


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
        if value.numel == 0:
            return torch.tensor([], dtype=value.dtype)
        else:
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
    if create and size == 0:
        logger.warning("Cannot create the shared memory with size = 0.")
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
            logger.info(
                f"The old size is {shm.size} and "
                f"create a new memory buffer with size {size}."
            )
            shm.unlink()
            shm = SharedMemory(
                name=name,
                create=create,
                size=size,
            )
    return shm


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
    if value.numel() == 0:
        return
    with torch.no_grad():
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
        meta_name = CheckpointSharedObjPrefix.META_NAME + str(local_rank)
        job_name = os.getenv(NodeEnv.TORCHELASTIC_RUN_ID, "")
        if job_name:
            self._shm_name = (
                job_name
                + "_"
                + CheckpointSharedObjPrefix.SHM_NAME
                + str(local_rank)
            )
        else:
            self._shm_name = CheckpointSharedObjPrefix.SHM_NAME + str(
                local_rank
            )
        self.shared_memory: Optional[SharedMemory] = None
        self.metadata = SharedDict(name=meta_name, create=host)
        self._need_creation = True

    def close(self):
        if self.shared_memory:
            self.shared_memory.close()

    def unlink(self):
        if not self.shared_memory:
            # The shared memory may be created by other processes.
            self.init_shared_memory()
        if self.shared_memory:
            self.shared_memory.unlink()
        if self.metadata:
            self.metadata.unlink()

    def reset(self):
        self._need_creation = True

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

    def save_state_dict(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """
        if not self.shared_memory:
            meta_dict = _traverse_state_dict(
                state_dict, self._create_tensor_meta
            )
            self.init_shared_memory(create=True, size=self._buffer_size)
        else:
            meta_dict = self.metadata.get(local=True)
        ckpt_conf: CheckpointConfig = meta_dict[DLROVER_CKPT_CONFIG_KEY]
        ckpt_conf.writing_shm = True

        self.metadata.set(meta_dict)
        assert self.shared_memory is not None
        _traverse_copy_to_shm(state_dict, meta_dict, self.shared_memory.buf)
        ckpt_conf.writing_shm = False
        self.metadata.set(meta_dict)

    def load_state_dict(self):
        """
        Load the state dict from the shared memory.

        Returns:
            Tuple(int, dict): The first value is the iteration step,
                the second value is the state dict.
        """
        meta_dict = self.metadata.get()
        default_config = CheckpointConfig()
        config = meta_dict.get(DLROVER_CKPT_CONFIG_KEY, default_config)
        if not meta_dict or config.writing_shm:
            return {}
        if self.shared_memory is None or self._need_creation:
            self.init_shared_memory(create=False)
        if not self.shared_memory:
            return {}

        state_dict = _read_state_dict_from_shm(meta_dict, self.shared_memory)
        return state_dict

    def no_checkpoint_state(self):
        """
        The handler lazily initializes the shared memory. The shared memory
        of the handler on the host may be None even if the handler on the
        device has saved state dict.
        """
        meta_dict = self.metadata.get()
        config: CheckpointConfig = meta_dict.get(DLROVER_CKPT_CONFIG_KEY, None)
        if config is None or config.step == 0:
            return True
        return False

    def init_shared_memory(self, create=False, size=0):
        self.shared_memory = _create_shared_memory(
            self._shm_name, create=create, size=size
        )
        self._need_creation = False

    def get_checkpoint_config(self, default_config):
        """
        Get the configuration of checkpointing state dict in the shared
        memory.

        Returns:
            A CheckpointShardConfig instance.
        """
        meta_dict = self.metadata.get()
        config = meta_dict.get(DLROVER_CKPT_CONFIG_KEY, default_config)
        return config


class AsyncCheckpointSaver(metaclass=ABCMeta):
    """
    CheckpointSaver asynchronously saves the state dict from the shared memory
    into the storage.

    Arguments:
        checkpoint_dir (str): the directory to save the checkpointing state
            dict to the storage if the training process fails.
        storage_meta (tuple[str]): the first element is the module path of the
            storage class and the second element is the name of
            the storage class.
        local_shard_num (int): the number of model/optimizer shards
            on the node.
        global_shard_num (int): the number of model/optimizer shards
            across all nodes.
    """

    _saver_instance = None
    _STAGE_DIR = "._dlrover_ckpt_stage"

    def __init__(
        self,
        checkpoint_dir,
        storage_meta: ClassMeta,
        local_shard_num=1,
        global_shard_num=1,
        save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.local_shard_num = local_shard_num
        self.global_shard_num = global_shard_num
        self._node_rank = env_utils.get_node_rank()
        self._is_agent_rank_0 = self._node_rank == 0
        self._shm_handlers: List[SharedMemoryHandler] = []
        self._shm_locks: List[SharedLock] = []
        self._stop_commit = False
        self._save_timeout = save_timeout

        module = importlib.import_module(storage_meta.module_path)
        storage_class_def = getattr(module, storage_meta.class_name)
        self.storage = storage_class_def(**storage_meta.kwargs)

        # Indicate whether the saver is writing state to storage
        self._writing_storage = False
        # The latest step to save the checkpoint.
        self._latest_step = 0
        qname = CheckpointSharedObjPrefix.SAVE_STEP_QNAME + str(0)
        self._event_queue = SharedQueue(name=qname, create=True)
        for i in range(self.local_shard_num):
            self._shm_handlers.append(SharedMemoryHandler(i))
            lock_name = CheckpointSharedObjPrefix.SHM_LOCK_NAME + str(i)
            self._shm_locks.append(SharedLock(name=lock_name, create=True))
        self._executor = ThreadPoolExecutor(
            max_workers=self.local_shard_num, thread_name_prefix="ckpt_saver-"
        )
        self._master_client = None

        # remove the history temp path if exists
        self.storage.safe_rmtree(
            os.path.join(self.checkpoint_dir, self._STAGE_DIR)
        )
        logger.info(
            "Initialize the AsyncSaver with arguments: "
            f"checkpoint_dir={checkpoint_dir}, "
            f"local_shard_num={local_shard_num}, "
            f"global_shard_num={global_shard_num}, "
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

        def _saver(class_meta: ClassMeta):

            # if the thread is not alive, the saver may be created
            if cls._saver_instance is not None:
                cls._saver_instance.close()
                cls._saver_instance = None

            module = importlib.import_module(class_meta.module_path)
            class_def = getattr(module, class_meta.class_name)

            saver: AsyncCheckpointSaver = class_def(**class_meta.kwargs)
            cls._saver_instance = saver
            cls._saver_instance._sync_shm_to_storage()

        sq = SharedQueue(name="factory", create=True)

        def _factory():
            logger.info("Start the checkpoint saver factory.")

            saver_thread = None
            while True:
                class_meta: ClassMeta = sq.get()

                # use a lock to avoid concurrent creation of the saver
                with (threading.Lock()):

                    # if the saver thread is alive, skip creating the saver
                    if (
                        cls._saver_instance
                        and saver_thread
                        and saver_thread.is_alive()
                    ):
                        logger.info(
                            "The saver is already created, "
                            "skip creating the saver."
                        )
                        continue

                    saver_thread = threading.Thread(
                        target=_saver,
                        args=(class_meta,),
                        name="checkpoint-saver",
                        daemon=True,
                    )
                    saver_thread.start()

        threading.Thread(
            target=_factory, name="checkpoint-saver-factory", daemon=True
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

    def get_master_client(self):
        if not self._master_client:
            self._master_client = MasterClient.singleton_instance()
            logger.info(f"Setup master client: {self._master_client}.")
        return self._master_client

    def wait_saving_checkpoint(self):
        """
        Check whether the saver finishes writing the
        latest checkpoint to the storage.
        """
        return self._writing_storage

    def close(self):
        """Clear the resource of the shared objects."""
        event = CheckpointEvent(type=CheckpointEventType.EXIT)
        if not self._event_queue.empty():
            self._event_queue.queue.get()
        self._event_queue.put(event)
        for i in range(self.local_shard_num):
            if self._shm_handlers[i]:
                self._shm_handlers[i].close()
                self._shm_handlers[i].unlink()
            self._shm_locks[i].unlink()
        self._event_queue.unlink()
        self._executor.shutdown(wait=False)

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """

        logger.info("Async flash checkpoint saver starts!")
        event: CheckpointEvent = None
        while True:
            try:
                event = self._event_queue.get()
                if event.type == CheckpointEventType.UPDATE_SHARD:
                    logger.info(
                        "Reset the shared memory after the training starts. "
                        "The number of global shards "
                        f"is {event.global_shard_num}."
                    )
                    self.global_shard_num = event.global_shard_num
                elif event.type == CheckpointEventType.SAVE:
                    logger.info(
                        "ShardingSaver save checkpoint to storage, "
                        f"event {event}"
                    )
                    self.save_step_checkpoint(event.step)
                elif event.type == CheckpointEventType.EXIT:
                    break
            except Exception as e:
                logger.error(
                    f"Got unexpected exception during checkpointing: {event}, "
                    f"error: {e}.",
                    exc_info=True,
                )
                self._report_failure_to_master(str(e))

    def _report_failure_to_master(self, error_msg):
        master_client = self.get_master_client()
        if not master_client:
            logger.warning(
                "Skip ckpt saver failure reporting for master "
                "client hasn't setup."
            )
            return

        if not error_msg:
            error_msg = "Unknown"
        error_full_msg = "Async checkpoint saver got failure:" + error_msg

        try:
            error = ProcessError(
                self._node_rank,
                -1,
                error_full_msg,
                datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            )

            master_client.report_failures(
                json.dumps(error),
                level=TrainingExceptionLevel.PROCESS_ERROR,
            )
        except Exception as e:
            logger.warning(
                "Failed to report failure to master " f"in ckpt saver: {e}."
            )

    def reset_shared_memory(self):
        self._stop_commit = True
        for shm_handler in self._shm_handlers:
            shm_handler.reset()

    def _save_shard(
        self,
        step,
        local_shard_id: int,
        ckpt_config: CheckpointConfig,
        step_done_dir: str,
    ):
        """Save the shard of state dict into the storage."""
        try:
            shm_handler = self._shm_handlers[local_shard_id]
            shm_lock = self._shm_locks[local_shard_id]
            if shm_handler.shared_memory is None:
                shm_handler.init_shared_memory(create=False)

            shm_lock.acquire()
            default_config = CheckpointConfig()
            config = shm_handler.get_checkpoint_config(default_config)
            if config.step != step:
                shm_lock.release()
                logger.error(
                    f"The step {step} in event is no equal "
                    f"to step {config.step} in memory."
                )
                return

            logger.info(
                f"Saves the checkpoint shard {local_shard_id} "
                f"of rank {ckpt_config.rank} from the "
                f"shared memory into the storage {ckpt_config}."
            )
            self.persist_to_storage(local_shard_id, ckpt_config)
            shm_lock.release()
            step_done_file = os.path.join(step_done_dir, str(ckpt_config.rank))
            self.storage.write("done", step_done_file)
            logger.info(
                f"Finish saving the checkpoint shard {local_shard_id} of "
                f"rank {ckpt_config.rank}."
            )
            return True
        except Exception as e:
            logger.error(
                f"Fail to save the checkpoint shard {local_shard_id} "
                f"of rank {ckpt_config.rank}, error: {e}",
                exc_info=True,
            )
            return False
        finally:
            shm_lock.release()

    def _dist_make_dir(self, path, timeout=30):
        if self._node_rank == 0:
            logger.info(f"Create path by rank0 worker: {path}.")
            self.storage.safe_rmtree(path)
            self.storage.safe_makedirs(path)
        else:
            for _ in range(timeout):
                if self.storage.exists(path):
                    break
                time.sleep(1)
            logger.warning(
                f"Worker {self._node_rank} can't find path {path} "
                f"with timeout {timeout}."
            )

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
        return done_dir

    def _check_shard_step_consistence(self, step, timeout=15):
        start_time = time.time()
        while True:
            steps = []
            for i in range(self.local_shard_num):
                default_config = CheckpointConfig()
                ckpt_config = self._shm_handlers[i].get_checkpoint_config(
                    default_config
                )
                if ckpt_config.step > 0:
                    steps.append(ckpt_config.step)
            if all(i == step for i in steps):
                return True
            time.sleep(1)
            if time.time() - start_time > timeout:
                logger.info(f"The cached steps are {steps}")
                return False

    def save_shm_to_storage(self, timeout=60, master_client=None):
        """
        Save the state dict in the shared memory into the storage. The agent
        can call the method to save the state dict into the storage if the
        training process fails or the agent wants to restart training
        processes.
        """
        if any(
            [handler.no_checkpoint_state() for handler in self._shm_handlers]
        ):
            logger.info(
                "Skip because no any memory buffer with the state dict."
            )
            return

        # Skip saving checkpoint if the step of checkpoint shard in the
        # memory is not same.
        steps = []
        for shm_handler in self._shm_handlers:
            default_config = CheckpointConfig()
            config = shm_handler.get_checkpoint_config(default_config)
            steps.append(config.step)
        if len(set(steps)) > 1:
            logger.error(
                "Skip because steps in shards are not "
                f"inconsistent: {steps}"
            )
            return

        step = steps[0]
        if master_client:
            # If some nodes failed, the alive nodes do not need to save its
            # checkpoint shards because some shards of failed node are missing.
            synced = self._sync_node_checkpoint(master_client, step, timeout)
            if not synced:
                logger.info(
                    "Skip saving the checkpoint from "
                    "the memory to the storage."
                )
                self._stop_commit = True
                return

        # The training process does not release the lock because it fails
        # when writing the state dict into the shared memory. The shared
        # memory may be dirty and the saver cannot save it to the storage.
        if self._writing_storage or self._any_rank_locked():
            logger.info(
                "Saver is writing the checkpoint to storage "
                "and skips saving at the breakpoint."
            )
            return

        if step > self._latest_step:
            self.save_step_checkpoint(step)
            logger.info(
                "Save the checkpointing state dict from the shared "
                f"memory to storage, step: {step}."
            )
        else:
            logger.info(f"The checkpoint of step {step} has been saved.")

    def _sync_node_checkpoint(
        self, master_client: MasterClient, step: int, timeout: int
    ):
        """
        Check whether all training node can save the checkpoint from the memory
        to the storage. If some nodes fail, other nodes needs not to save
        """
        start = time.time()
        while True:
            success = master_client.sync_checkpoint(step)
            if success:
                return success
            else:
                time.sleep(3)
                elapsed_time = time.time() - start
                if elapsed_time > timeout:
                    logger.info(
                        "It is timeout to sync checkpoint "
                        "bacause some nodes may fail."
                    )
                    return False

    @classmethod
    def reset(cls):
        """Reset the shared memory of all shards."""
        if cls._saver_instance is None:
            return
        cls._saver_instance.reset_shared_memory()
        logger.info("Reset all shared memory of shards.")

    @abstractmethod
    def save_step_checkpoint(self, step: int):
        """
        Save the checkpoint of a step into the storage.

        Args:
            step (int): the iteration step.
        """
        pass

    @abstractmethod
    def persist_to_storage(self, local_shard_id, ckpt_config):
        """
        Persist the state dict to a storage path.

        Args:
            local_shard_id (int): the index of local shard.
            ckpt_config : the checkpoint config with the path to
                save the storage.
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
        checkpoint state dict is saved to the storage.

        Args:
            step (int): the iteration step.
        """
        pass


class CommonDirCheckpointSaver(AsyncCheckpointSaver):
    """
    The saver asynchronously save the checkpointing state dict from
    the shared memory to the storage. It launches threads to directly save
    each shard into the path configured by users. The thread saves each shard
    to a unique path and writes a done file after the saving is finished.
    The path is generated by the training framework like Megatron-LM,
    DeepSpeed or users' definition. The saver updates the tracer file with
    the step if the number of done file is equal to the number of shard.
    """

    def update_tracker_file(self, step):
        """
        Write the step into the tracker file.

        Args:
            step (int): the checkpointing step.
        """
        tracker_filename = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        self.storage.write(str(step), tracker_filename)

    def save_step_checkpoint(self, step: int):
        """
        Save the checkpoint of a step into the storage.

        Args:
            step (int): the iteration step.
        """
        passed = self._check_shard_step_consistence(step)
        if not passed:
            logger.warning(
                f"Skip persisting the checkpoint of step {step} "
                "because the cached step in memory are not consistent."
            )
            return
        self._writing_storage = True

        step_done_dir = self._get_checkpoint_done_dir(step)
        self._dist_make_dir(step_done_dir)

        write_success = False
        # save to stage path for each local rank
        futures: List[Future] = []
        for i in range(self.local_shard_num):
            default_config = CheckpointConfig()
            ckpt_config = self._shm_handlers[i].get_checkpoint_config(
                default_config
            )
            if ckpt_config.step == 0:
                continue
            future: Future = self._executor.submit(
                self._save_shard,
                step,
                i,
                ckpt_config,
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
            self._stop_commit = False
            self.commit_checkpoint(
                step, step_done_dir, timeout=self._save_timeout
            )

        self._writing_storage = False

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
        success = False
        while True:
            if self._stop_commit:
                logger.info(
                    "Stop committing the checkpoint because "
                    "the training processes restarted."
                )
                break
            done_files = self.storage.listdir(step_done_dir)
            ready_num = len(done_files)
            if ready_num == self.global_shard_num:
                logger.info(
                    f"All agents finish saving checkpoint for step {step}"
                )
                self.update_tracker_file(step)
                # clean stage dir
                self.storage.safe_rmtree(step_done_dir)
                success = True
                break
            logger.info(
                f"The number of ready shards is {ready_num} "
                f"!= {self.global_shard_num}."
            )
            # timeout
            elapsed_time = round(time.time() - start_time, 2)
            if elapsed_time > timeout:
                logger.error(
                    f"Commit checkpoint timeout for step {step}, "
                    f"elapsed_time: {elapsed_time}. The done files "
                    f"are {done_files}."
                )
                # clean stage dir
                self.storage.safe_rmtree(step_done_dir)
                break

            time.sleep(5)
        self.storage.commit(step, success)

    def persist_to_storage(
        self, local_shard_id: int, ckpt_config: CheckpointConfig
    ):
        state_dict = self._shm_handlers[local_shard_id].load_state_dict()
        for state_name, sd in state_dict.items():
            if sd and state_name in ckpt_config.paths:
                path = ckpt_config.paths[state_name]
                self.storage.write_state_dict(sd, path, torch.save)


class TempDirCheckpointSaver(AsyncCheckpointSaver):
    """
    This saver firstly saves the checkpoint shards into the temporary
    directory and move the directory into the common directory configured
    by users.
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
        passed = self._check_shard_step_consistence(step)
        if not passed:
            logger.warning(
                f"Skip persisting the checkpoint of step {step} "
                "because the cached step in memory are not consistent."
            )
            return
        self._writing_storage = True
        mkdir_timeout = int(self._save_timeout / 2)

        temp_dir = self._get_tmp_ckpt_dir(step)
        self._dist_make_dir(temp_dir, mkdir_timeout)
        step_done_dir = self._get_checkpoint_done_dir(step)
        self._dist_make_dir(step_done_dir, mkdir_timeout)

        write_success = False
        # save to stage path for each local rank
        futures: List[Future] = []
        ckpt_dir = ""
        for i in range(self.local_shard_num):
            default_config = CheckpointConfig()
            ckpt_config = self._shm_handlers[i].get_checkpoint_config(
                default_config
            )
            ckpt_dir = self._replace_path_dir(ckpt_config, temp_dir)
            future = self._executor.submit(
                self._save_shard, step, i, ckpt_config, step_done_dir
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
                tmp_path=temp_dir,
                target_path=ckpt_dir,
                timeout=self._save_timeout,
            )

        self._writing_storage = False

    def _replace_path_dir(self, ckpt_config: CheckpointConfig, temp_dir: str):
        """
        Replace the directory with the temp directory.

        Returns:
            str: the original directory.
        """
        ckpt_dir = ""
        if not ckpt_config.paths:
            return ckpt_dir
        tmp_paths = {}
        for name, path in ckpt_config.paths.items():
            path = str(path)
            path_dir = os.path.dirname(path)
            tmp_paths[name] = path.replace(path_dir, temp_dir)
            if not ckpt_dir:
                ckpt_dir = path_dir
            elif not fu.is_same_path(ckpt_dir, path_dir):
                raise ValueError(
                    "The directories must be same. The latest dir "
                    f"is {ckpt_dir} and the current dir  of {name} "
                    f"is {path_dir}"
                )
        ckpt_config.paths = tmp_paths
        return ckpt_dir

    def _get_tmp_ckpt_dir(self, step: int):
        """
        Get the temp directory to save the latest checkpoint. After all
        shards are saved, the saver will move the directory to a
        regular directory defined by users.
        """
        ckpt_name = os.path.join(
            self.checkpoint_dir, self._STAGE_DIR, str(step)
        )
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
            step_done_dir (str): the directory to save the done file of
                each shard.
            tmp_path: the temp directory path to save the latest checkpoint.
            target_path: the regular directory path to save the checkpoint.
            timeout (int): the timeout to wait all shards are saved,
                default 600s.
        """
        logger.info(
            f"Start commit checkpoint tmp_path: {tmp_path}, "
            f"path: {target_path}"
        )
        start_time = time.time()
        success = False
        while True:
            done_files = self.storage.listdir(step_done_dir)
            ready_num = len(done_files)
            # Check whether all shards are completed.
            if ready_num == self.global_shard_num:
                logger.info(
                    f"All agents finish saving checkpoint for step {step}"
                )

                if os.path.exists(target_path):
                    if os.path.isdir(target_path):
                        self.storage.safe_rmtree(target_path)
                    else:
                        self.storage.safe_remove(target_path)

                # commit checkpoint
                self.storage.safe_move(tmp_path, target_path)
                # clean stage dir
                self.storage.safe_rmtree(step_done_dir)
                self.update_tracker_file(step)
                logger.info(
                    f"Commit checkpoint tmp_path: {tmp_path}, "
                    f"path: {target_path}"
                )
                success = True
                break

            logger.info(
                f"The number of ready shards is {ready_num} "
                f"!= {self.global_shard_num}."
            )

            # timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error(
                    f"Commit checkpoint timeout for step {step}, "
                    f"elapsed_time: {elapsed_time}. The done files "
                    f"are {done_files}."
                )
                # clean stage dir
                self.storage.safe_rmtree(tmp_path)
                self.storage.safe_rmtree(step_done_dir)
                break

            time.sleep(5)
        self.storage.commit(step, success)


class DdpCheckpointSaver(CommonDirCheckpointSaver):
    """Persist the checkpoint from CPU memory buffer into the storage."""

    def persist_to_storage(self, local_shard_id: int, ckpt_config):
        if self._node_rank != 0:
            logger.info("Skip and only rank 0 saves checkpoint in a DDP job.")
            return
        super().persist_to_storage(local_shard_id, ckpt_config)


class MegatronCheckpointSaver(CommonDirCheckpointSaver):
    TRACER_FILE = "latest_checkpointed_iteration.txt"

    def update_tracker_file(self, step):
        """
        Write the step into the tracker file.

        Args:
            step (int): the checkpointing step.
        """
        tracker_filename = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        self.storage.write(str(step), tracker_filename)
        tracker_filename = os.path.join(self.checkpoint_dir, self.TRACER_FILE)
        self.storage.write(str(step), tracker_filename)


class DeepSpeedCheckpointSaver(CommonDirCheckpointSaver):
    TRACER_FILE = "latest"

    def update_tracker_file(self, step):
        """
        Write the step into the tracker file.

        Args:
            step (int): the checkpointing step.
        """
        tracker_filename = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        self.storage.write(str(step), tracker_filename)
        ds_tracker_filename = os.path.join(
            self.checkpoint_dir, self.TRACER_FILE
        )
        self.storage.write(str(step), ds_tracker_filename)


class FsdpDcpSaver(CommonDirCheckpointSaver):
    """The saver saves the distributed checkpoint of FSDP into the storage."""

    def persist_to_storage(
        self,
        local_shard_id: int,
        ckpt_config: CheckpointConfig,
    ):
        """
        Persist the state dict to a storage path.

        Args:
            local_shard_id (int): the index of local shard.
            ckpt_config : the checkpoint config with the path to
                save the storage.
        """

        shm_handler = self._shm_handlers[local_shard_id]
        path = ckpt_config.paths[CheckpointConstant.MODEL_STATES_NAME]
        checkpoint_dir = os.path.dirname(path)

        # only rank0 create dir
        if self._is_agent_rank_0 and local_shard_id == 0:
            self._dist_make_dir(checkpoint_dir)
        else:
            while not self.storage.exists(checkpoint_dir):
                time.sleep(1)

        # do saving
        assert shm_handler.shared_memory is not None
        self.storage.write(shm_handler.shared_memory.buf, path)

        # operate meta
        if self._is_agent_rank_0 and local_shard_id == 0:
            parent_path = Path(os.path.dirname(path))
            meta_dict = shm_handler.metadata.get()
            dcp_metadata = meta_dict.get("dcp_metadata", {})
            if dcp_metadata:
                data = pickle.dumps(dcp_metadata)
                self.storage.write(data, parent_path / ".metadata")
            tracer_file = os.path.join(
                self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
            )
            self.storage.write(str(ckpt_config.step), tracer_file)
