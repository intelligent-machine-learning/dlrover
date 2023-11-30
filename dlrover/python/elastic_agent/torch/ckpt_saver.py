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
import shutil
import sys
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Mapping, Tuple

import numpy as np
import torch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.shared_obj import (
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
)

CKPT_DIR_PREFIX = "checkpoint-"

SAVE_STEP_QNAME_PREFIX = "checkpoint_lock_rank_"
CKPT_META_NAME_PREFIX = "checkpoint_meta_local_rank_"
TENSOR_SHM_NAME_PREFIX = "checkpoint_shm_local_rank_"
SHM_LOCK_NAME_PREFIX = "shm_local_rank_"


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


def convert_torch_dtype_to_numpy(torch_dtype):
    dtype_map = {
        torch.float32: np.float32,
        torch.float: np.float32,
        torch.float64: np.float64,
        torch.double: np.double,
        torch.float16: np.float16,
        torch.half: np.half,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.short: np.short,
        torch.int32: np.int32,
        torch.int: np.int32,
        torch.long: np.int64,
        torch.bool: np.dtype("bool"),
    }
    return dtype_map[torch_dtype]


def traverse_state_dict(value: object, visitor: Callable[[object], None]):
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    """
    if isinstance(value, Mapping):
        temp_dict = {}
        for k, v in value.items():
            temp_dict[k] = traverse_state_dict(v, visitor)
        return temp_dict
    elif isinstance(value, List):
        temp_list = []
        for _, v in enumerate(value):
            temp_list.append(traverse_state_dict(v, visitor))
        return temp_list
    else:
        return visitor(value)


def read_state_dict_from_shm(checkpoint_meta, tensor_shm):
    state_dict = traverse_state_dict(
        checkpoint_meta,
        lambda x: read_tensor_from_buf(x, tensor_shm),
    )
    return state_dict


def read_tensor_from_buf(value, shm_tensor_buffer):
    """
    Read a tensor from the buffer of shared memory.
    """
    if isinstance(value, TensorMeta):
        data_array = np.frombuffer(
            buffer=shm_tensor_buffer.buf,
            dtype=value.dtype,
            offset=value.offset,
            count=value.numel,
        )
        value = torch.reshape(torch.tensor(data_array), value.shape)
        return value
    else:
        return value


@dataclass
class TensorMeta(object):
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


class SaverFactory(object):
    """
    Save the checkpointing state dict from the shared memory
    into the storage.
    """

    pass


class CheckpointSaver(metaclass=ABCMeta):
    @abstractmethod
    def _save_shm_to_storage(self):
        pass

    @classmethod
    def start_async_saving_ckpt(cls):
        """
        Start a thread to asynchronously save the checkpoint state dict
        from the shared memory into the storage. Firstly, it waits that
        the training process notify the saver class to create a saver.
        """
        sq = SharedQueue(name="factory", create=True)

        def _save(sq: SharedQueue):
            class_name = sq.get()
            class_def = getattr(sys.modules[__name__], class_name)
            saver: CheckpointSaver = class_def()
            saver._save_shm_to_storage()

        threading.Thread(
            target=_save, args=(sq,), name="checkpoint-saver", daemon=True
        ).start()


class NoShardingSaver(CheckpointSaver):
    """
    The saver only saves the state dict without sharding
    from the shared memory created by local rank 0 to the storage.
    """

    def __init__(self) -> None:
        self._checkpoint_dir = ""
        self._tensor_shm = None
        # Only local rank 0 save the state dict to memory in DDP.
        qname = SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=True)
        meta_name = CKPT_META_NAME_PREFIX + str(0)
        self._shared_ckpt_meta = SharedDict(name=meta_name, recv=True)
        lock_name = SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=True)
        self._shm_name = TENSOR_SHM_NAME_PREFIX + str(0)

    def __del__(self):
        if self._tensor_shm:
            self._tensor_shm.close()
            self._tensor_shm.unlink()

    def _save_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Start saving the checkpointing state dict to storage.")
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
            state_dict = read_state_dict_from_shm(meta_dict, self._tensor_shm)
            self._persist_to_storage(state_dict, path)
            self._shm_lock.release()

    def _persist_to_storage(self, state_dict, path):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        checkpoint_dir = os.path.dirname(path)
        _init_dir(checkpoint_dir)
        torch.save(state_dict, path)
