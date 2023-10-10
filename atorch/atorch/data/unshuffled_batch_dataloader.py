import math

import torch.distributed as dist
from torch.utils.data.distributed import Sampler


class DistributedUnshuffledBatchSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if batch_size is None:
            raise RuntimeError("Requires batch_size to be available")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = []
        batch_num = int(self.num_samples / self.batch_size)
        for i in range(batch_num):
            start_pos = self.rank * self.batch_size + self.num_replicas * self.batch_size * i
            end_pos = start_pos + self.batch_size
            indices.extend(range(start_pos, end_pos))
        return iter(indices)

    def __len__(self):
        return self.num_samples
