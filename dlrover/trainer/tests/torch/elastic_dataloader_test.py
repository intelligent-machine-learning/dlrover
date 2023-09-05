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
from unittest.mock import patch

import numpy as np
from torch.utils.data import Dataset

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

    @patch("builtins.open", create=True)
    def test_load_config(self, mock_open):
        dataset = SimpleDataset()
        # Create a temporary ElasticDataLoader instance for testing
        dataloader = ElasticDataLoader(dataset=dataset, batch_size=32)

        # Assert that the loaded batch_size is correct
        self.assertEqual(dataloader.current_batch_size, 32)

        # Configure the mock_open to return the desired content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"batch_size": 64}'
        )

        # Call the load_config method
        dataloader.load_config(config_file="config.json")

        # Configure the mock_open to return the desired content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"batch_size": 128}'
        )

        dataloader = ElasticDataLoader(
            dataset=dataset, config_file="config.json"
        )

        # Assert that the loaded batch_size is correct
        self.assertEqual(dataloader.current_batch_size, 128)

    @patch("builtins.open", create=True)
    def test_set_batch_size(self, mock_open):
        dataset = SimpleDataset()
        # Configure the mock_open to return the desired content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"batch_size": 128}'
        )
        dataloader = ElasticDataLoader(
            dataset=dataset, batch_size=32, config_file="config.json"
        )

        # Call the set_batch_size method to change the batch_size
        dataloader.set_batch_size(128)

        # Assert that the set batch_size is correct
        self.assertEqual(dataloader.current_batch_size, 128)
