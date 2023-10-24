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

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from dlrover.python.common.log import default_logger as logger
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler

CKPT_DIR_PREFIX = "checkpoint-"


def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def sync():
    if dist.is_initialized():
        dist.barrier()


def get_latest_checkpoint(checkpoint_dir):
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


class CheckpointManger(metaclass=ABCMeta):
    """CheckpontManager can save and load checkpoint states.

    Args:
        model (nn.Module): an instance of `torch.nn.Module`.
        optimizer (Optimizer): an instance of `torch.optim.Optimizer`.
        dataloader (DataLader): an instance of `torch.utils.data.DataLoader`.
            The sampler of DataLoader should be an instance of
            `dlrover.trainer.torch.elastic.ElasticDistribuedSampler`.
        directory (str): the directory to save the checkpoint states.
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
        directory,
        rank=0,
        max_to_keep=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.directory = directory
        self.rank = rank
        self.max_to_keep = max_to_keep

    def log_rank0(self, log):
        if self.rank == 0:
            logger.info(log)

    def _is_rank0(self):
        return self.rank == 0

    def _keep_topk_checkpoint(self):
        """Keep top k checkpoints and remove other checkpoints."""
        if not self.max_to_keep or not self._is_rank0():
            return
        steps = []
        for dir_name in os.listdir(self.directory):
            if not dir_name.startswith(CKPT_DIR_PREFIX):
                continue
            step = int(dir_name.split("-")[-1])
            steps.append(step)

        steps = sorted(steps)
        if len(steps) <= self.max_to_keep:
            return
        remove_steps = steps[: -1 * self.max_to_keep]
        for step in remove_steps:
            dir_name = os.path.join(self.directory, f"{CKPT_DIR_PREFIX}{step}")
            shutil.rmtree(dir_name)

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
    def load(self, ckpt_path=None):
        """
        The manager loads the states from the files in the
        checkpoint direcotry to the model, optimizer and sampler.

        ckpt_path (str, optinoal): The manager will load checkpoint from the
            path. If the path is None, the manager will load the state
            checkpoint from the file with the maximum step.
        """
        pass

    @classmethod
    def init_checkpoint_manager(
        cls, model, optimizer, dataloader, directory, rank=0, max_to_keep=None
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
                rank,
                max_to_keep,
            )
        elif isinstance(model, DDP):
            return DDPCheckpointManger(
                model,
                optimizer,
                dataloader,
                directory,
                rank,
                max_to_keep,
            )
        elif isinstance(model, FSDP):
            return FSDPCheckpointManger(
                model,
                optimizer,
                dataloader,
                directory,
                rank,
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
        ckpt_dir = os.path.join(self.directory, f"{CKPT_DIR_PREFIX}{step}")
        init_dir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        torch.save(checkpoint, ckpt_path)
        self._keep_topk_checkpoint()

    def load(self, ckpt_path=None):
        latest_ckpt_dir = get_latest_checkpoint(self.directory)
        if not latest_ckpt_dir:
            return
        if not ckpt_path:
            ckpt_path = os.path.join(latest_ckpt_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            return
        logger.info(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)


class DDPCheckpointManger(LocalCheckpointManger):
    """DDPCheckpontManager saves and loads checkpoint states of a DDP model.

    Example::
        >>> ckpt_manager = CheckpointManager(
        >>>    model, optimizer, train_dataloader, "/tmp/checkpoint/"
        >>> )
        >>> ckpt_manager.save(0, 10)
        >>> ckpt_manger.load()
    """

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
        ckpt_dir = os.path.join(self.directory, f"{CKPT_DIR_PREFIX}{step}")
        if self._is_rank0():
            init_dir(ckpt_dir)
        sync()
        # Only rank0 saves the checkpoint for DDP model.
        if self._is_rank0():
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            torch.save(checkpoint, ckpt_path)
        self._keep_topk_checkpoint()
        sync()


class FSDPCheckpointManger(CheckpointManger):
    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/`. All ranks will save
        the part of the model and optimizer states into the file
        `checkpoint-{step}/part-{rank}.pt`.
        """
        self.log_rank0(f"Save checkpoint of step={step} of epoch={epoch}.")
        step = step + epoch * len(self.dataloader)
        FSDP.set_state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        )
        msd = self.model.state_dict()
        osd = FSDP.optim_state_dict(self.model, self.optimizer)
        ssd = {}
        if isinstance(self.dataloader.sampler, ElasticDistributedSampler):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
        checkpoint = {"model": msd, "optimizer": osd, "sampler": ssd}
        ckpt_dir = os.path.join(self.directory, f"{CKPT_DIR_PREFIX}{step}")
        if self._is_rank0():
            init_dir(ckpt_dir)
        sync()
        ckpt_path = os.path.join(ckpt_dir, f"part-{self.rank}.pt")
        torch.save(checkpoint, ckpt_path)
        self._keep_topk_checkpoint()
        sync()

    def load(self, ckpt_path=None):
        latest_ckpt_dir = get_latest_checkpoint(self.directory)
        if not latest_ckpt_dir:
            return
        if not ckpt_path:
            ckpt_path = os.path.join(latest_ckpt_dir, f"part-{self.rank}.pt")
        if not os.path.exists(ckpt_path):
            return
        logger.info(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        optim_state_dict = checkpoint.get("optimizer", {})

        FSDP.set_state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        )
        self.model.load_state_dict(model_state_dict)

        # called from all ranks, though only rank0 has
        # a valid param for full_osd.
        optim_state_dict = FSDP.optim_state_dict_to_load(
            model=self.model,
            optim=self.optimizer,
            optim_state_dict=optim_state_dict,
        )
        self.optimizer.load_state_dict(optim_state_dict)
        sync()
