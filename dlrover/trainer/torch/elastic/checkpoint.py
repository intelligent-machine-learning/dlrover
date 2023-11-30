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
import time
from abc import ABCMeta, abstractmethod
from datetime import timedelta
from typing import List, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.shared_obj import (
    SharedDict,
    SharedLock,
    SharedMemory,
    SharedQueue,
)
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CKPT_META_NAME_PREFIX,
    SAVE_STEP_QNAME_PREFIX,
    SHM_LOCK_NAME_PREFIX,
    TENSOR_SHM_NAME_PREFIX,
    TensorMeta,
    convert_torch_dtype_to_numpy,
    read_state_dict_from_shm,
    traverse_state_dict,
)
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler

CKPT_DIR_PREFIX = "checkpoint-"


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(f"Function {func.__name__} cost {t}s")
        return result

    return wrapper


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
    if max_step > 0:
        path = os.path.join(checkpoint_dir, f"{CKPT_DIR_PREFIX}{max_step}")
    else:
        path = ""
    return path


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


class CheckpointEngine(metaclass=ABCMeta):
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
            self._saver_group = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=30)
            )
        else:
            self._rank = 0
            self._local_rank = 0
            self._saver_group = None

        self._buffer_size = 0
        self._cached_step = 0
        self._meta_dict = dict()
        self._shm_name = ""
        self._tensor_shm: SharedMemory = None
        self._shared_ckpt_meta: SharedDict = None
        self._shm_buffer_lock: SharedLock = None
        self._to_save_queue: SharedQueue = None
        self._notify_agent_to_create_saver()
        self._init_shared_objs()

    def __del__(self):
        self.close()

    def close(self):
        if self._shared_ckpt_meta:
            self._shared_ckpt_meta.close()
        if self._shm_buffer_lock:
            self._shm_buffer_lock.close()
        if self._to_save_queue:
            self._to_save_queue.close()
        if self._tensor_shm:
            self._tensor_shm.close()

    @abstractmethod
    def _init_shared_objs(self):
        """
        Initialize the shared queue, lock and memory to communiate
        with the agent in the main process.
        """
        pass

    @abstractmethod
    def _notify_agent_to_create_saver(self):
        """
        Notify the agent in the main process to create a checkpointing
        saver to save the state dict from the shared memory into the storage.
        """
        pass

    def _create_tensor_meta(self, value: torch.Tensor):
        """
        Create a tensor meta of a tensor and compute the total
        size of the state dict.
        """
        if not torch.is_tensor(value):
            return value
        dtype = convert_torch_dtype_to_numpy(value.dtype)
        meta = TensorMeta(
            shape=tuple(value.shape),  # type: ignore
            dtype=dtype,
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
        self._meta_dict = traverse_state_dict(
            state_dict, self._create_tensor_meta
        )

        # Update the meta dict in the main process.
        self._shared_ckpt_meta.update(self._meta_dict)
        self._tensor_shm = _create_shared_memory(
            name=self._shm_name,
            create=True,
            size=self._buffer_size,
        )

    def _copy_state_dict_to_shm(self, state_dict):
        """
        Copy the state dict from CPU memory buffer into the shared memory.
        """

        def _tarverse_copy(value, meta):
            if isinstance(value, Mapping):
                for k, v in value.items():
                    if isinstance(v, (Mapping, List)):
                        m = meta[k]
                        _tarverse_copy(v, m)
                    elif torch.is_tensor(v):
                        m = meta[k]
                        self._write_shared_memory(v, m)
                    else:
                        meta[k] = v
            elif isinstance(value, List):
                for i, v in enumerate(value):
                    if isinstance(v, (Mapping, List)):
                        m = meta[i]
                        _tarverse_copy(v, m)
                    elif torch.is_tensor(v):
                        m = meta[i]
                        self._write_shared_memory(v, m)
                    else:
                        meta[i] = v

        _tarverse_copy(state_dict, self._meta_dict)
        # Update the meta dict in the main process.
        self._shared_ckpt_meta.update(self._meta_dict)

    def _write_shared_memory(self, value, meta: TensorMeta):
        """
        Write a CPU tensor into the shared memory.
        """
        data_array = value.cpu().numpy()
        write_array = np.ndarray(
            data_array.shape,
            dtype=data_array.dtype,
            buffer=self._tensor_shm.buf,
            offset=meta.offset,
        )
        if data_array.shape == ():
            write_array.fill(data_array)
        else:
            write_array[:] = data_array[:]

    @timer
    def save_to_memory(self, state_dict, step):
        """
        Synchonously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
        """
        state_dict["step"] = step
        if self._tensor_shm is None:
            self._make_state_dict_buffer(state_dict)
        acquired = self._shm_buffer_lock.acquire(blocking=False)
        all_rank_ready = self._check_all_rank_ready(acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_buffer_lock.release()
            return
        self._copy_state_dict_to_shm(state_dict)

        if acquired:
            self._shm_buffer_lock.release()
        self._cached_step = step

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

    @timer
    def save_to_storage(self, state_dict, path, step):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
            step (int): the iteration step.
        """
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
        if self._tensor_shm is None:
            self._tensor_shm = _create_shared_memory(
                self._shm_name,
                create=False,
            )
        if not self._tensor_shm:
            return None
        meta_dict = self._shared_ckpt_meta.get()
        state_dict = read_state_dict_from_shm(meta_dict, self._tensor_shm)
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
            state_dict = self._load_from_historic_checkpoint()
        return state_dict

    def _load_from_historic_checkpoint(self):
        """Locd checkpoint from the lastest complete checkpoint."""
        while True:
            latest_ckpt_dir = _get_latest_checkpoint(self.checkpoint_dir)
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


class ShardedCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine to save the sharded model and optimizer state dict
    into the memory and storage. We can use it to save the model and optimizer
    using FSDP, Zero-3 or Megatron-LM.
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir)

    def _notify_agent_to_create_saver(self):
        queue = SharedQueue(name="factory")
        queue.put("ShardingSaver")
        queue.close()

    def _init_shared_objs(self):
        meta_name = CKPT_META_NAME_PREFIX + str(self._local_rank)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=False)
        lock_name = SHM_LOCK_NAME_PREFIX + str(self._local_rank)
        self._shm_buffer_lock = SharedLock(name=lock_name, create=False)
        qname = SAVE_STEP_QNAME_PREFIX + str(self._local_rank)
        self._to_save_queue = SharedQueue(name=qname, create=False)
        self._shm_name = TENSOR_SHM_NAME_PREFIX + str(self._local_rank)


class NoShardingCheckpointEngine(CheckpointEngine):
    """
    The `save` of the engine only writes the state dict into the shared memory.
    A subprocess will asychronously save the state dict into the storage.
    Writing to memory is significantly quicker than writing to storage.
    The engine.save only block the training with a little time.

    Attributes:
        checkpoint_dir: str, the directory to save the checkpoint.
        save_storage_interval: int, the interval of iteration steps to save
            the model and optimizer states from CPU memory to the storage.
        max_to_keep: int, the number of checkpoint files to keep.

    Examples::
        >>> engine = NoShardingCheckpointEngine(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>>     save_storage_interval=5,
        >>>     max_to_keep=1,
        >>> )
        >>> state_dict = model.state_dict()
        >>> engine.save(step=100, state_dict=state_dict)
        >>> engine.wait()
        >>> sate_dict = engine.load()
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir)

    def _notify_agent_to_create_saver(self):
        queue = SharedQueue(name="factory")
        queue.put("NoShardingSaver")
        queue.close()

    def _init_shared_objs(self):
        """
        Initialize the shared object with the main process.
        Without model sharding, all ranks share the same shared memory
        created by the local rank 0 on a node.
        """
        meta_name = CKPT_META_NAME_PREFIX + str(0)
        self._shared_ckpt_meta = SharedDict(name=meta_name, create=False)
        lock_name = SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_buffer_lock = SharedLock(name=lock_name, create=False)
        qname = SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=False)
        self._shm_name = TENSOR_SHM_NAME_PREFIX + str(0)

    @timer
    def save_to_memory(self, state_dict, step):
        """
        Synchonously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
        """
        if self._local_rank == 0:
            super().save_to_memory(state_dict, step)

    @timer
    def save_to_storage(self, state_dict, path, step):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
        """
        if self._rank == 0:
            super().save_to_storage(state_dict, path, step)


class CheckpointManger(metaclass=ABCMeta):
    """CheckpontManager can save and load checkpoint states.

    Args:
        model (nn.Module): an instance of `torch.nn.Module`.
        optimizer (Optimizer): an instance of `torch.optim.Optimizer`.
        dataloader (DataLader): an instance of `torch.utils.data.DataLoader`.
            The sampler of DataLoader should be an instance of
            `dlrover.trainer.torch.elastic.ElasticDistribuedSampler`.
        checkpoint_dir (str): the directory to save the checkpoint states.
        save_storage_interval (int, optinal): The step inverval to save the
            checkoint state dict into the storage. Default: ``1``.
        max_to_keep (int, optinal): the max number of checkpoint to keep. The
            oldest checkpoint files will be removed if the number of
            checkpoints is bigger than max_to_kep. Default: ``1``.

    Example::
        >>> ckpt_manager = LocalCheckpointManger(
        >>>    model=model,
        >>>    optimizer=optimizer,
        >>>    dataloader=train_dataloader,
        >>>    checkpoint_dir="/tmp/checkpoint/",
        >>>    save_storage_interval=5,
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
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir
        self.save_storage_interval = save_storage_interval
        self.max_to_keep = max_to_keep
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self._rank = 0
            self._local_rank = 0

    def _log_rank0(self, log):
        if self._rank == 0:
            logger.info(log)

    def _engine_save(self, engine: CheckpointEngine, state_dict, step):
        """
        The each rank has the complete state dict without sharding. Only
        the locak rank 0 on each node saves the state dict into the shared
        memory and only the rank 0 saves the state dict into the storage.
        """
        engine.save_to_memory(state_dict, step)
        if step % self.save_storage_interval == 0:
            if self._rank == 0:
                _keep_topk_checkpoint(
                    self.checkpoint_dir, self.max_to_keep - 1
                )
            ckpt_dir = os.path.join(
                self.checkpoint_dir, f"{CKPT_DIR_PREFIX}{step}"
            )
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            engine.save_to_storage(state_dict, ckpt_path, step)

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
    """
    The manager saves and loads checkpoint states of the local
    model and optimizer without distributed execution.
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
        super().__init__(
            model,
            optimizer,
            dataloader,
            checkpoint_dir,
            save_storage_interval,
            max_to_keep,
        )
        self._ckpt_engine = NoShardingCheckpointEngine(
            checkpoint_dir,
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
        checkpoint = {
            "model": msd,
            "optimizer": osd,
            "sampler": ssd,
            "epoch": epoch,
        }
        self._engine_save(self._ckpt_engine, checkpoint, step)

    def load(self, resuming_path=None):
        """
        Load teh state dict from checkpointing data to the model and optimizer.
        """
        checkpoint = self._ckpt_engine.load(resuming_path)
        if not checkpoint:
            return
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)


class DDPCheckpointManger(LocalCheckpointManger):
    """
    DDPCheckpontManager saves and loads checkpoint states of a DDP model.
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
        super().__init__(
            model,
            optimizer,
            dataloader,
            checkpoint_dir,
            save_storage_interval,
            max_to_keep,
        )

    def load(self, resuming_path=None):
        """
        Load teh state dict from checkpointing data to the model and optimizer.
        """
        super().load(resuming_path=resuming_path)
        _sync()


class FSDPCheckpointManger(CheckpointManger):
    """
    DDPCheckpontManager saves and loads checkpoint states of a DDP model.
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
        super().__init__(
            model,
            optimizer,
            dataloader,
            checkpoint_dir,
            save_storage_interval,
            max_to_keep,
        )
        self._ckpt_engine = NoShardingCheckpointEngine(checkpoint_dir)

    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/`. All ranks will save
        the part of the model and optimizer states into the file
        `checkpoint-{step}/part-{rank}.pt`.
        """
        self._log_rank0(f"Save checkpoint of step={step} of epoch={epoch}.")
        if self.dataloader:
            step = step + epoch * len(self.dataloader)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
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
        checkpoint = {
            "model": msd,
            "optimizer": osd,
            "sampler": ssd,
            "epoch": epoch,
        }
        self._engine_save(self._ckpt_engine, checkpoint, step)

    def load(self, resuming_path=None):
        """
        Load teh state dict from checkpointing data to the model and optimizer.
        """
        checkpoint = self._save_engine.load(resuming_path)
        if not checkpoint:
            return
        if self.dataloader:
            sampler = self.dataloader.sampler
            if isinstance(sampler, ElasticDistributedSampler):
                sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})

        #  TODO: use shard_state_dict to checkpoint.
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
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
