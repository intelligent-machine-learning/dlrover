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

import multiprocessing
import os
import shutil
import time
import random
import string
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Callable, List, Mapping, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP

from dlrover.python.common.log import default_logger as logger
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler


CKPT_DIR_PREFIX = "checkpoint-"


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


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


def _sync():
    if dist.is_initialized():
        dist.barrier()


def _get_latest_checkpoint(checkpoint_dir):
    """Get the checkpoint directory with the maximum step."""
    if not os.path.exists(checkpoint_dir):
        return ""
    max_step = 0
    for fn in os.listdir(checkpoint_dir):
        if not fn.startswith(CKPT_DIR_PREFIX):
            continue
        step = int(fn.split("-")[-1])
        max_step = step if step > max_step else max_step
    path = os.path.join(checkpoint_dir, f"{CKPT_DIR_PREFIX}{max_step}")
    return path


def _keep_topk_checkpoint(checkpoint_dir, max_to_keep):
    """Keep top k checkpoints and remove other checkpoints.

    Arguments:
        checkpoint_dir: the directory to save checkpoint files.
        max_to_keep: the number of checkpoint files to keep.
    """
    steps = []
    for dir_name in os.listdir(checkpoint_dir):
        if not dir_name.startswith(CKPT_DIR_PREFIX):
            continue
        step = int(dir_name.split("-")[-1])
        steps.append(step)

    steps = sorted(steps)
    if len(steps) <= max_to_keep:
        return
    remove_steps = steps[: -1 * max_to_keep]
    for step in remove_steps:
        dir_name = os.path.join(checkpoint_dir, f"{CKPT_DIR_PREFIX}{step}")
        shutil.rmtree(dir_name)


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


@dataclass
class TensorMeta(object):
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


class CheckpointManger(metaclass=ABCMeta):
    """CheckpontManager can save and load checkpoint states.

    Args:
        model (nn.Module): an instance of `torch.nn.Module`.
        optimizer (Optimizer): an instance of `torch.optim.Optimizer`.
        dataloader (DataLader): an instance of `torch.utils.data.DataLoader`.
            The sampler of DataLoader should be an instance of
            `dlrover.trainer.torch.elastic.ElasticDistribuedSampler`.
        checkpoint_dir (str): the directory to save the checkpoint states.
        rank (int): the rank of process in the communication world.
        max_to_keep (int): the max number of checkpoint to keep. The oldest
            checkpoint files will be removed if the number of checkpoints
            is bigger than max_to_kep.
    """

    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        checkpoint_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def log_rank0(self, log):
        if self.rank == 0:
            logger.info(log)

    def _is_rank0(self):
        return self.rank == 0

    @abstractmethod
    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer and sampler.

        Args:
            epoch (int): the epoch index.
            step (int): the iteration step in the epoch.
        """
        pass

    @abstractmethod
    def load(self, resuming_path=None):
        """
        The manager loads the states from the files in the
        checkpoint direcotry to the model, optimizer and sampler.

        resuming_path (str, optinoal): The manager will load checkpoint from
            the path. If the path is None, the manager will load the state
            checkpoint from the file with the maximum step.
        """
        pass

    @classmethod
    def init_checkpoint_manager(
        cls,
        model,
        optimizer,
        dataloader,
        directory,
        max_to_keep=1,
        save_storage_interval=1,
    ):
        """A factory method to initialize a checkpoint manager by the model
        class.
        """
        if not dist.is_initialized():
            return LocalCheckpointManger(
                model,
                optimizer,
                dataloader,
                directory,
                save_storage_interval,
                max_to_keep,
            )
        elif isinstance(model, DDP):
            return DDPCheckpointManger(
                model,
                optimizer,
                dataloader,
                directory,
                save_storage_interval,
                max_to_keep,
            )
        elif isinstance(model, FSDP):
            return FSDPCheckpointManger(
                model,
                optimizer,
                dataloader,
                directory,
                save_storage_interval,
                max_to_keep,
            )
        else:
            raise NotImplementedError(f"Not support model class {model}")


class LocalCheckpointManger(CheckpointManger):
    """The manager saves and loads checkpoint states of the local
    model and optimizer without distributed execution.

    Example::
        >>> ckpt_manager = LocalCheckpointManger(
        >>>    model, optimizer, train_dataloader, "/tmp/checkpoint/"
        >>> )
        >>> ckpt_manager.save(0, 10)
        >>> ckpt_manger.load()

    """

    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        checkpoint_dir,
        save_storage_interval,
        max_to_keep=1,
    ):
        super().__init__(model, optimizer, dataloader, checkpoint_dir)
        self._save_engine = LocalAsyncCkptEngine(
            checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/checkpoint.pt`.
        """
        logger.info(f"Save checkpoint of step={step} of epoch={epoch}.")
        step = step + epoch * len(self.dataloader)
        msd = self.model.state_dict()
        osd = self.optimizer.state_dict()
        ssd = {}
        if isinstance(self.dataloader.sampler, ElasticDistributedSampler):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
        checkpoint = {"model": msd, "optimizer": osd, "sampler": ssd}
        self._save_engine.save(step, checkpoint)

    def load(self, resuming_path=None):
        checkpoint = self._save_engine.load(resuming_path)
        if not checkpoint:
            return
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)


class DDPCheckpointManger(CheckpointManger):
    """DDPCheckpontManager saves and loads checkpoint states of a DDP model.

    Example::
        >>> ckpt_manager = CheckpointManager(
        >>>    model, optimizer, train_dataloader, "/tmp/checkpoint/"
        >>> )
        >>> ckpt_manager.save(0, 10)
        >>> ckpt_manger.load()
    """

    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        super().__init__(model, optimizer, dataloader, checkpoint_dir)
        self._save_engine = DDPAsyncCkptEngine(
            checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/checkpoint.pt`.
        """
        self.log_rank0(f"Save checkpoint of step={step} of epoch={epoch}.")
        step = step + epoch * len(self.dataloader)
        msd = self.model.state_dict()
        osd = self.optimizer.state_dict()
        ssd = {}
        if isinstance(self.dataloader.sampler, ElasticDistributedSampler):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
        checkpoint = {"model": msd, "optimizer": osd, "sampler": ssd}
        self._save_engine.save(step, checkpoint)

    def load(self, resuming_path=None):
        checkpoint = self._save_engine.load(resuming_path)
        if not checkpoint:
            return
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)
        _sync()


class FSDPCheckpointManger(CheckpointManger):
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        super().__init__(model, optimizer, dataloader, checkpoint_dir)
        self._save_engine = LocalAsyncCkptEngine(
            checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/`. All ranks will save
        the part of the model and optimizer states into the file
        `checkpoint-{step}/part-{rank}.pt`.
        """
        self.log_rank0(f"Save checkpoint of step={step} of epoch={epoch}.")
        if self.dataloader:
            step = step + epoch * len(self.dataloader)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
        ):
            msd = self.model.state_dict()
            osd = FSDP.optim_state_dict(self.model, self.optimizer)
        ssd = {}
        if self.dataloader and isinstance(
            self.dataloader.sampler, ElasticDistributedSampler
        ):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
        checkpoint = {"model": msd, "optimizer": osd, "sampler": ssd}
        self._save_engine.save(checkpoint)

    def load(self, resuming_path=None):
        checkpoint = self._save_engine.load(resuming_path)
        if not checkpoint:
            return
        if self.dataloader:
            sampler = self.dataloader.sampler
            if isinstance(sampler, ElasticDistributedSampler):
                sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})

        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
        ):
            # called from all ranks, though only rank0 has
            # a valid param for full_osd.
            optim_state_dict = FSDP.optim_state_dict_to_load(
                model=self.model,
                optim=self.optimizer,
                optim_state_dict=optim_state_dict,
            )
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)
        _sync()


class AsyncCheckpointEngine(metaclass=ABCMeta):
    """
    Attributes:
        checkpoint_dir: str, the directory to save the checkpoint.
        max_to_keep: int, the number of checkpoint files to keep.
        save_storage_interval: int, the interval of iteration steps to save
            the model and optimizer states from CPU memory to the storage.
    """

    def __init__(
        self,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.save_storage_interval = save_storage_interval
        self._manager = multiprocessing.Manager()
        self._tensor_meta_buffer = self._manager.dict()
        self._memory_buffer = None
        self._shm_tensor_buffer = None
        self._shm_buffer_lock = multiprocessing.Lock()
        self._buffer_size = 0
        self._checkpoint_step_queue = multiprocessing.Queue(maxsize=1)
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._saver_group = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=30)
            )
        else:
            self._rank = 0
            self._saver_group = None
        random_name = get_random_string(8)
        self._shm_name = f"tensor_buffer_{random_name}_{self._rank}"
        self._persist_thread = multiprocessing.Process(
            name=f"persist-process-rank-{self._rank}",
            target=self._persist_memory_buffer_to_storage,
            daemon=True,
        )
        self._persist_thread.start()
        self._check_arguments()

    def _check_arguments(self):
        if self.max_to_keep == 0:
            raise ValueError("max_to_keep cannot be 0.")
        if self.save_storage_interval == 0:
            raise ValueError("save_storage_interval cannot be 0.")

    def _allocate_tensor_memory(self, value):
        if not torch.is_tensor(value):
            return value
        pin_memory = False if value.device.type == "cpu" else True
        t = torch.empty_like(value.cpu(), pin_memory=pin_memory)
        return t

    def _create_tensor_meta(self, value):
        """
        Create a tensor meta of a tensor and compute the total
        size of the state dict.
        """
        if not torch.is_tensor(value):
            return value
        meta = TensorMeta(
            shape=tuple(value.shape),
            dtype=value.numpy().dtype,
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
        self._memory_buffer = traverse_state_dict(
            state_dict, self._allocate_tensor_memory
        )
        meta_dict = traverse_state_dict(
            self._memory_buffer, self._create_tensor_meta
        )
        self._tensor_meta_buffer.update(meta_dict)
        self._shm_tensor_buffer = shared_memory.SharedMemory(
            create=True,
            size=self._buffer_size,
            name=self._shm_name,
        )

    def _copy_state_dict_to_shm(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """

        def _tarverse_copy(origin_value, target_value, meta):
            if isinstance(origin_value, Mapping):
                for k, ov in origin_value.items():
                    if isinstance(ov, (Mapping, List)):
                        tv = target_value[k]
                        m = meta[k]
                        _tarverse_copy(ov, tv, m)
                    elif torch.is_tensor(ov):
                        tv = target_value[k]
                        tv.copy_(ov)
                        m = meta[k]
                        self._write_shared_memory(tv, m)
                    else:
                        target_value[k] = ov
            elif isinstance(origin_value, List):
                for i, ov in enumerate(origin_value):
                    if isinstance(ov, (Mapping, List)):
                        tv = target_value[i]
                        m = meta[i]
                        _tarverse_copy(ov, tv, m)
                    elif torch.is_tensor(ov):
                        tv = target_value[i]
                        tv.copy_(ov)
                        m = meta[i]
                        self._write_shared_memory(tv, m)
                    else:
                        target_value[i] = ov

        _tarverse_copy(
            state_dict, self._memory_buffer, self._tensor_meta_buffer
        )

    def _write_shared_memory(self, value, meta: TensorMeta):
        """
        Write a CPU tensor into the shared memory.
        """
        data_array = value.numpy()
        write_array = np.ndarray(
            data_array.shape,
            dtype=data_array.dtype,
            buffer=self._shm_tensor_buffer.buf,
            offset=meta.offset,
        )
        if data_array.shape == ():
            write_array.fill(data_array)
        else:
            write_array[:] = data_array[:]

    def _persist_memory_buffer_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Start the process to persist the state dict.")
        shm_tensor_buffer = None
        while True:
            step = self._checkpoint_step_queue.get()
            if not shm_tensor_buffer:
                shm_tensor_buffer = shared_memory.SharedMemory(
                    name=self._shm_name,
                )
            with self._shm_buffer_lock:
                checkpoint_dir = os.path.join(
                    self.checkpoint_dir, f"{CKPT_DIR_PREFIX}{step}"
                )
                logger.info(
                    f"Save step-{step} checkpoint from  memory "
                    f"into the storage {checkpoint_dir}."
                )
                meta_dict = {}
                meta_dict.update(self._tensor_meta_buffer)
                state_dict = traverse_state_dict(
                    meta_dict,
                    lambda x: self._read_tensor_from_buf(x, shm_tensor_buffer),
                )
                self._persist_to_storage(state_dict, checkpoint_dir)

    def _read_tensor_from_buf(self, value, shm_tensor_buffer):
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

    @abstractmethod
    def _persist_to_storage(self, state_dict, checkpoint_dir):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        pass

    @timer
    def save(self, step, state_dict):
        """
        Save the state dict into the CPU memory. If the step is the multiple
        of the save_storage_interval, the engine will persist the state dict
        from the CPU memory into the storage.

        Args:
            step: the iteration step in the training loop.
            state_dict: a dictionary.
        """
        state_dict["step"] = step
        if self._shm_tensor_buffer is None:
            self._make_state_dict_buffer(state_dict)
        acquired = self._shm_buffer_lock.acquire(block=False)
        all_rank_ready = self._check_all_rank_ready(acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint with "
                f"step {step} in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_buffer_lock.release()
            return
        self._copy_state_dict_to_shm(state_dict)
        if step % self.save_storage_interval == 0:
            self._checkpoint_step_queue.put(step)
        if acquired:
            self._shm_buffer_lock.release()

    def _check_all_rank_ready(self, ready):
        """
        Check wether all ranks are ready.
        """
        if not self._saver_group:
            return ready
        value = 0 if ready else 1
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, group=self._saver_group)
        return t == 0

    @abstractmethod
    def load(self, resume_path=""):
        """
        Load the state dict from the CPU memory if the state dict is complete
        in CPU memory. Otherwise, the function will load the state dict from
        the storage.

        Args:
            resume_path: str, If the resume_path is an empty
                string, the function will load the latest checkpoint file in
                the checkpoint directory.

        Returns:
            A dict.
        """
        pass


class LocalAsyncCkptEngine(AsyncCheckpointEngine):
    def __init__(
        self,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    @timer
    def _persist_to_storage(self, state_dict, checkpoint_dir):
        _init_dir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(state_dict, checkpoint_path)
        _keep_topk_checkpoint(self.checkpoint_dir, self.max_to_keep)

    def load(self, resuming_path=""):
        if not resuming_path:
            latest_ckpt_dir = _get_latest_checkpoint(self.checkpoint_dir)
            if not latest_ckpt_dir:
                return {}
            resuming_path = os.path.join(latest_ckpt_dir, "checkpoint.pt")

        if not os.path.exists(resuming_path):
            return {}
        logger.info(f"Load checkpoint from {resuming_path}")
        checkpoint = torch.load(resuming_path)
        return checkpoint


class DDPAsyncCkptEngine(LocalAsyncCkptEngine):
    def __init__(
        self,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    @timer
    def _persist_to_storage(self, state_dict, checkpoint_dir):
        if self._rank == 0:
            _init_dir(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(state_dict, checkpoint_path)
            _keep_topk_checkpoint(self.checkpoint_dir, self.max_to_keep)


class FSDPAsyncCkptEngine(AsyncCheckpointEngine):
    def __init__(
        self,
        checkpoint_dir,
        save_storage_interval=1,
        max_to_keep=1,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            save_storage_interval=save_storage_interval,
            max_to_keep=max_to_keep,
        )

    @timer
    def _persist_to_storage(self, state_dict, checkpoint_dir):
        if self._rank == 0:
            _init_dir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"part-{self._rank}.pt")
        torch.save(state_dict, checkpoint_path)
        _keep_topk_checkpoint(self.checkpoint_dir, self.max_to_keep)

    def load(self, resuming_path=None):
        if not resuming_path:
            latest_ckpt_dir = _get_latest_checkpoint(self.checkpoint_dir)
            if not latest_ckpt_dir:
                return
            resuming_path = os.path.join(
                latest_ckpt_dir, f"part-{self._rank}.pt"
            )
        if not os.path.exists(resuming_path):
            return {}
        logger.info(f"Load checkpoint from {resuming_path}")
        checkpoint = torch.load(resuming_path)
        return checkpoint
