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

import unittest

import numpy as np
from torch.utils.data import Dataset

from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 60001)

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

        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
        )
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)

        for i in sampler:
            pass
        sampler.set_epoch(1)
        val = next(iter(sampler))
        self.assertEqual(val, 0)

        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)
