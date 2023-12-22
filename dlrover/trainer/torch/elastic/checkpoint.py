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
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP

from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.log import default_logger as logger
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler
from dlrover.trainer.torch.flash_checkpoint.ddp_engine import (
    DdpCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.engine import CheckpointEngine

CKPT_DIR_PREFIX = "checkpoint-"


def _sync():
    if dist.is_initialized():
        dist.barrier()


def _keep_topk_checkpoint(checkpoint_dir, max_to_keep):
    """Keep top k checkpoints and remove other checkpoints.

    Arguments:
        checkpoint_dir: the directory to save checkpoint files.
        max_to_keep: the number of checkpoint files to keep.
    """
    step_names: Dict[int, str] = {}
    if not os.path.exists(checkpoint_dir):
        return
    for ckpt_name in os.listdir(checkpoint_dir):
        if not ckpt_name.startswith(
            CheckpointConstant.CKPT_NAME_PREFIX
        ) or not ckpt_name.endswith(".pt"):
            continue
        name = ckpt_name.split("-")[-1]
        if name.endswith(".pt"):
            step = int(name.split(".")[0])
        else:
            step = int(name)
        step_names[step] = ckpt_name

    steps = sorted(list(step_names.keys()))
    if len(steps) <= max_to_keep:
        return
    if max_to_keep == 0:
        remove_steps = steps
    else:
        remove_steps = steps[: -1 * max_to_keep]
    for step in remove_steps:
        ckpt_name = os.path.join(checkpoint_dir, step_names[step])
        logger.info(f"Remove the checkpoint {ckpt_name}")
        if os.path.isfile(ckpt_name):
            os.remove(ckpt_name)
        else:
            shutil.rmtree(ckpt_name)


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
            self._local_rank = int(os.getenv("LOCAL_RANK", 0))

    def _log_rank0(self, log):
        if self._rank == 0:
            logger.info(log)

    def _engine_save(self, engine: CheckpointEngine, step, state_dict):
        """
        The each rank has the complete state dict without sharding. Only
        the locak rank 0 on each node saves the state dict into the shared
        memory and only the rank 0 saves the state dict into the storage.
        """
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"{CheckpointConstant.CKPT_NAME_PREFIX}{step}.pt",
        )
        engine.save_to_memory(step, state_dict, ckpt_path)
        if step % self.save_storage_interval == 0:
            if self._rank == 0:
                _keep_topk_checkpoint(
                    self.checkpoint_dir, self.max_to_keep - 1
                )
            engine.save_to_storage(step, state_dict, ckpt_path)

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

        Args:
            resuming_path (str, optinoal): The manager will load checkpoint
                from the path. If the path is None, the manager will load
                the state checkpoint from the file with the maximum step.

        Return:
            step (int): the iteration step.
            A dict: a state dict.
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
        self._ckpt_engine = DdpCheckpointEngine(
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
            "step": step,
        }
        self._engine_save(self._ckpt_engine, step, checkpoint)

    def load(self, resuming_path=None):
        """
        Load teh state dict from checkpointing data to the model and optimizer.
        """
        checkpoint = self._ckpt_engine.load(resuming_path)
        if not checkpoint:
            return {}
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)
        return checkpoint


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
        checkpoint = super().load(resuming_path=resuming_path)
        _sync()
        return checkpoint


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
        self._ckpt_engine = DdpCheckpointEngine(checkpoint_dir)

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
            StateDictType.SHARDED_STATE_DICT,
        ):
            state_dict = {
                "model": self.model.state_dict(),
                "optim": FSDP.optim_state_dict(self.model, self.optimizer),
            }

        ssd = {}
        if self.dataloader and isinstance(
            self.dataloader.sampler, ElasticDistributedSampler
        ):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
            state_dict["sampler"] = ssd
        state_dict["epoch"] = epoch
        state_dict["step"] = step
        subdir_name = CheckpointConstant.CKPT_NAME_PREFIX + str(step)
        checkpoint_dir = os.path.join(self.checkpoint_dir, subdir_name)
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(checkpoint_dir),
        )
        tracer_file = os.path.join(
            self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
        )
        with open(tracer_file, "w") as f:
            f.write(str(step))

    def load(self, resuming_path=None):
        """
        Load teh state dict from checkpointing data to the model and optimizer.
        """

        if resuming_path is None:
            tracer_file = os.path.join(
                self.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
            )
            if not os.path.exists(tracer_file):
                return {}
            with open(tracer_file, "r") as f:
                step = f.read()
                subdir_name = CheckpointConstant.CKPT_NAME_PREFIX + step
                resuming_path = os.path.join(self.checkpoint_dir, subdir_name)
        with FSDP.state_dict_type(
            self.model, StateDictType.SHARDED_STATE_DICT
        ):
            # cannot load the optimizer state_dict together
            # with the model state_dict.
            state_dict = {
                "model": self.model.state_dict(),
                "step": 0,
                "epoch": 0,
                "sampler": {},
            }

            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(resuming_path),
            )
            self.model.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=dist_cp.FileSystemReader(resuming_path),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                self.model, self.optimizer, optim_state["optim"]
            )
            self.optimizer.load_state_dict(flattened_osd)

        if self.dataloader:
            sampler = self.dataloader.sampler
            if isinstance(sampler, ElasticDistributedSampler):
                sampler.load_state_dict(state_dict.get("sampler", {}))
        return state_dict
