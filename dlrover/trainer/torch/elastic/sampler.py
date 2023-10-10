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

from dlrover.python.common.log import default_logger as logger

T_co = TypeVar("T_co", covariant=True)


class ElasticDistributedSampler(DistributedSampler):
    """ElasticDistributedSampler can checkpoint unused sample indices
    and restore sample indices from the checkpoint to support
    fault-tolerance.

    Example::

    >>> dataset = torchvision.datasets.ImageFolder(
    ...     root=args.training_data,
    ...     transform=transforms.ToTensor(),
    ... )
    >>> sampler = ElasticDistributedSampler(dataset=dataset)
    >>> dataloader = DataLoader(
    ...     dataset=train_data,
    ...     batch_size=args.batch_size,
    ...     num_workers=2,
    ...     sampler=sampler,
    ... )
    >>> for epoch in range(start_epoch, n_epochs):
    ...     sampler.set_epoch(epoch)
    ...     train(dataloader)
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
        self._epoch_checkpoint: Dict[int, int] = {}

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
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        completed_num = self._epoch_checkpoint.get(self.epoch, 0)
        start_iter = self.rank + completed_num
        # fmt: off
        indices = indices[start_iter:self.total_size:self.num_replicas]
        # fmt: on
        if self.epoch not in self._epoch_checkpoint:
            self._init_num_samples()
        assert len(indices) == self.num_samples

        return iter(indices)

    def _init_num_samples(self):
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

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
        return state

    def load_state_dict(self, state: Dict[str, int]):
        """
        Restore the uncompleted shards from a checkpoint. The shard
        client will send uncompleted shards to the DLRover job master.
        The master will assign those shards to workers to restore training.
        """
        self.epoch = int(state.get("epoch", 0))
        completed_num = int(state.get("completed_num", 0))
        dataset_size = len(self.dataset)
        if completed_num > dataset_size:
            completed_num = completed_num % dataset_size
        remaining_samples = dataset_size - completed_num
        self._epoch_checkpoint[self.epoch] = completed_num
        if self.drop_last and remaining_samples % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (remaining_samples - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(remaining_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas + completed_num
        logger.info(
            "Load epoch = %s, completed num = %s, num_samples = %s",
            self.epoch,
            completed_num,
            self.num_samples,
        )
