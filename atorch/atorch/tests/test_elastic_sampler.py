import unittest

import numpy as np
from torch.utils.data import Dataset

from atorch.data.elastic_sampler import ElasticDistributedSampler


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 60000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ElasticDistributedSamplerTest(unittest.TestCase):
    def test_checkpoint(self):
        dataset = SimpleDataset()
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )
        batch_size = 8
        step = 0
        sampler_state = None
        for i, v in enumerate(sampler):
            if i % batch_size == 0:
                step = i / batch_size
            if step == 4:
                sampler_state = sampler.state_dict(step, batch_size)
                break
        self.assertEqual(sampler_state["completed_num"], 64)
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)
