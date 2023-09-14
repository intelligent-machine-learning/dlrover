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
import tempfile
import unittest

import numpy as np
from torch.utils.data import Dataset

from dlrover.python.common.grpc import ParallelConfig
from dlrover.trainer.torch.elastic.dataloader import ElasticDataLoader


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 60000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestElasticDataLoader(unittest.TestCase):
    def setUp(self):
        # Initialization code to run before each test method
        pass

    def tearDown(self):
        # Cleanup code to run after each test method
        pass

    def test_load_config(self):
        dataset = SimpleDataset()
        # Create a temporary ElasticDataLoader instance for testing
        dataloader = ElasticDataLoader(dataset=dataset, batch_size=32)

        # Assert that the loaded batch_size is correct
        self.assertEqual(dataloader.batch_sampler.batch_size, 32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config = ParallelConfig()
            config.dataloader.batch_size = 128
            config.dataloader.version = 1
            config_file = os.path.join(tmpdirname, "config.json")
            with open(config_file, "w") as f:
                f.write(config.to_json())

            # Call the load_config method
            dataloader.load_config(config_file=config_file)
            dataloader = ElasticDataLoader(
                dataset=dataset, config_file=config_file
            )

            # Assert that the loaded batch_size is correct
            self.assertEqual(dataloader.batch_sampler.batch_size, 128)

            with open(config_file, "w") as f:
                f.write("")
            dataloader.load_config(config_file=config_file)

    def test_set_batch_size(self):
        dataset = SimpleDataset()
        config = ParallelConfig()
        config.dataloader.batch_size = 128
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "config.json")
            with open(config_file, "w") as f:
                f.write(config.to_json())

            dataloader = ElasticDataLoader(
                dataset=dataset, batch_size=32, config_file=config_file
            )

            # Call the set_batch_size method to change the batch_size
            dataloader.update_batch_size(128)

            # Assert that the set batch_size is correct
            self.assertEqual(dataloader.batch_sampler.batch_size, 128)
