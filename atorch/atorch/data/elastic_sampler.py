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
import math
from typing import Dict, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from atorch.common.log_utils import default_logger as logger

T_co = TypeVar("T_co", covariant=True)


class ElasticDistributedSampler(DistributedSampler):
    """ElasticDistributedSampler can checkpoint unused sample indices
    and restore sample indices from the checkpoint to support
    fault-tolerance. The detail of args is in
    https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

    Example:
    >>> train_data = torchvision.datasets.ImageFolder(
    >>>     root="mnist/training/",
    >>>     transform=transforms.ToTensor(),
    >>> )
    >>> #  Setup sampler for elastic training.
    >>> sampler = ElasticDistributedSampler(dataset=train_data)
    >>> train_loader = DataLoader(
    >>>     dataset=train_data,
    >>>     batch_size=16,
    >>>     num_workers=2,
    >>>     sampler=sampler,
    >>> )
    >>> ...
    >>> # Save checkpiont of sampler
    >>> sampler_state = train_loader.sampler.state_dict(
    >>> train_step, train_loader.batch_size
    >>> ),  # Checkpoint sampler
    >>> # load checkpoint of sampler
    >>> train_loader.sampler.load_state_dict(sampler_state)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not dist.is_initialized():
            rank = 0 if not rank else rank
            num_replicas = 1 if not num_replicas else num_replicas

        super(ElasticDistributedSampler, self).__init__(
            dataset,
            num_replicas,
            rank,
            shuffle,
            seed,
            drop_last,
        )
        self.completed_num = 0

    def __iter__(self) -> Iterator[T_co]:
        indices = []  # type: ignore
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        start_iter = self.rank + self.completed_num
        # fmt: off
        indices = indices[start_iter:self.total_size:self.num_replicas]
        # fmt: on
        assert len(indices) == self.num_samples

        return iter(indices)

    def state_dict(self, iter_step, micro_batch_size):
        """Checkpoint the index of the last completed sample.
        In DDP training, the completed number of sample of each
        step is the micro_batch_size * num_replicas.
        """
        completed_num = iter_step * micro_batch_size * self.num_replicas
        state = {
            "completed_num": completed_num,
            "epoch": self.epoch,
        }
        logger.info("ElasticDistributedSampler state: %s", state)
        return state

    def load_state_dict(self, state: Dict[str, int]):
        """
        Restore the uncompleted shards from a checkpoint. The shard
        client will send uncompleted shards to the DLRover job master.
        The master will assign those shards to workers to restore training.
        """
        self.epoch = int(state.get("epoch", 0))
        self.completed_num = int(state.get("completed_num", 0))
        self.num_samples = int((self.total_size - self.completed_num) / self.num_replicas)
        if self.completed_num > self.total_size:
            self.completed_num = self.completed_num % self.total_size
        logger.info(
            "Load epoch = %s, completed num = %s, num_samples = %s",
            self.epoch,
            self.completed_num,
            self.num_samples,
        )
