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
    def __init__(self, len=60001):
        self.data = np.arange(0, len)

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
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        sampler.load_state_dict(sampler_state)
        val = next(iter(sampler))
        self.assertEqual(val, 64)

    def test_checkpoint_with_scaling(self):
        dataset = SimpleDataset(len=60000)
        # 1 Train with 8 replicas, epoch is 0
        batch_size = 8
        step = 0
        checkpoint_step = 4
        num_replicas = 8
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            shuffle=False,
        )
        sampler.set_epoch(0)

        # 2 Save the checkpoint
        sampler_state = None
        val = 0
        for i, v in enumerate(sampler):
            self.assertEqual(val, v)
            val += num_replicas
            if i % batch_size == 0:
                step = i / batch_size
            if step == checkpoint_step:
                sampler_state = sampler.state_dict(step, batch_size)
                break
        self.assertEqual(
            sampler_state["completed_num"], 8 * batch_size * checkpoint_step
        )

        # 3 Resume with 6 replicas from checkpoint, and epoch is 0
        sampler.set_epoch(0)
        num_replicas = 6
        sampler = ElasticDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            shuffle=False,
        )
        sampler.load_state_dict(sampler_state)
        val = 8 * batch_size * checkpoint_step
        for i in sampler:
            self.assertEqual(val, i)
            val += num_replicas

        # 4 Continue, but epoch is 1
        sampler.set_epoch(1)
        val = 0
        for i in sampler:
            self.assertEqual(val, i)
            val += num_replicas
