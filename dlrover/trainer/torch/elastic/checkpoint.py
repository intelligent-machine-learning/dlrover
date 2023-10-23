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

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from dlrover.python.common.log import default_logger as logger
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler

CKPT_DIR_PREFIX = "checkpoint-"


def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


class CheckpointManger(object):
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
            is bigger than max_to_kepp.

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

    def save(self, epoch, step):
        """
        Save the checkpoint of model, optimizer, dataloader into the directory
        `{self.directory}/checkpoint-{step}/`. If use_fdsp, all ranks will save
        the part of the model and optimizer states into the file
        `checkpoint-{step}/part-{rank}.pt`. If not use_fsdp, only rank0 saves
        the total model and optimizer states into the file
        `checkpoint-{step}/checkpoint.pt`.

        Args:
            epoch (int): the epoch index.
            step (int): the iteration step in the epoch.
        """
        self.log_rank0(f"Save checkpoint of step={step} of epoch={epoch}.")
        step = step + epoch * len(self.dataloader)
        is_fsdp_model = isinstance(self.model, FSDP)
        msd, osd = get_model_optim_state(
            self.model, self.optimizer, is_fsdp_model
        )
        ssd = {}
        if isinstance(self.dataloader.sampler, ElasticDistributedSampler):
            ssd = self.dataloader.sampler.state_dict(
                step, self.dataloader.batch_size
            )
        checkpoint = {"model": msd, "optimizer": osd, "sampler": ssd}
        ckpt_dir = os.path.join(self.directory, f"{CKPT_DIR_PREFIX}{step}")
        if self._is_rank0():
            init_dir(ckpt_dir)
        if is_fsdp_model:
            ckpt_path = os.path.join(ckpt_dir, f"part-{self.rank}.pt")
            torch.save(checkpoint, ckpt_path)
        else:
            # Only rank0 saves the checkpoint for DDP model.
            if self._is_rank0():
                ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
                torch.save(checkpoint, ckpt_path)
        self._keep_topk_checkpoint()
        dist.barrier()

    def _keep_topk_checkpoint(self):
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

    def load(self, ckpt_path=None):
        """
        The manager firstly searches the checkpoint directory with the maximum
        step. Then, the manager loads the states from the files in the
        checkpoint direcotry to the model, optimizer and sampler.

        ckpt_path (str, optinoal): The manager will load checkpoint from the
            path. If the path is None, the manager will load the state
            checkpoint from the file with the maximum step.
        """
        is_fsdp_model = isinstance(self.model, FSDP)
        latest_ckpt_dir = get_latest_checkpoint(self.directory)
        if not latest_ckpt_dir:
            return
        if not ckpt_path:
            if is_fsdp_model:
                rank = dist.get_rank()
                ckpt_path = os.path.join(latest_ckpt_dir, f"part-{rank}.pt")
            else:
                ckpt_path = os.path.join(latest_ckpt_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            return
        logger.info(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        sampler = self.dataloader.sampler
        if isinstance(sampler, ElasticDistributedSampler):
            sampler.load_state_dict(checkpoint.get("sampler", {}))
        model_state_dict = checkpoint.get("model", {})
        self.model.load_state_dict(model_state_dict)
        optim_state_dict = checkpoint.get("optimizer", {})
        if is_fsdp_model:
            FSDP.set_state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
            )
            # called from all ranks, though only rank0 has
            # a valid param for full_osd.
            optim_state_dict = FSDP.optim_state_dict_to_load(
                optim_state_dict, self.model, self.optimizer
            )
            self.optimizer.load_state_dict(optim_state_dict)
        else:
            self.optimizer.load_state_dict(optim_state_dict)


def get_model_optim_state(model, optimizer, use_fsdp=False):
    """Get model and optimizer states."""
    if use_fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()
    return model_state, optim_state


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
