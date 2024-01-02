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

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data import Dataset

from dlrover.python.common.constants import ConfigPath
from dlrover.python.common.grpc import ParallelConfig
from dlrover.trainer.torch.elastic.dataloader import ElasticDataLoader
from dlrover.trainer.torch.elastic.trainer import (
    ElasticTrainer,
    _ElasticLRScheduler,
    _ElasticOptimizer,
)


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = np.arange(0, 60000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ElasticTrainerTest(unittest.TestCase):
    def setUp(self):
        self.model_mock = MagicMock()
        self.elastic_trainer = ElasticTrainer(self.model_mock)

    def test_step_context(self):
        model = torch.nn.Linear(10, 10)
        fsdp_trainer = ElasticTrainer(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer = self.elastic_trainer.prepare(optimizer)

        # Create dummy data for testing
        data = torch.rand(10, 10)

        with fsdp_trainer.step():
            output = model(data)
            loss = torch.sum(output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.assertTrue(self.elastic_trainer.gradient_state.sync_gradients)

        with fsdp_trainer.step():
            output = model(data)
            loss = torch.sum(output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.assertTrue(self.elastic_trainer.gradient_state.sync_gradients)

    def test_prepare_without_lr_scheduler(self):
        optimizer_mock = MagicMock()
        prepared_optimizer = self.elastic_trainer.prepare(optimizer_mock)
        self.assertIsInstance(prepared_optimizer, _ElasticOptimizer)

    def test_prepare_with_lr_scheduler(self):
        optimizer_mock = MagicMock()
        lr_scheduler_mock = MagicMock()
        prepared_optimizer, prepared_scheduler = self.elastic_trainer.prepare(
            optimizer_mock, lr_scheduler_mock
        )
        self.assertIsInstance(prepared_optimizer, _ElasticOptimizer)
        self.assertIsInstance(prepared_scheduler, _ElasticLRScheduler)

    def test_reset(self):
        self.elastic_trainer.reset()
        self.assertEqual(self.elastic_trainer.gradient_state.num_steps, 0)

    def test_report_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "runtime_metrics.json")
            os.environ[ConfigPath.ENV_RUNTIME_METRICS] = config_file
            self.elastic_trainer._last_report_time = 0
            self.elastic_trainer._after_step()
            with open(config_file, "r") as f:
                runtime_metrics = json.load(f)
                step = runtime_metrics.get("step")
                self.assertEqual(step, 1)

    def test_update_dataloader(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "paral_config.json")
            os.environ[ConfigPath.ENV_PARAL_CONFIG] = config_file
            dataset = SimpleDataset()
            dataloader = ElasticDataLoader(dataset=dataset, batch_size=32)
            model = torch.nn.Linear(10, 10)
            trainer = ElasticTrainer(model, dataloader=dataloader)
            config = ParallelConfig()
            config.dataloader.batch_size = 64
            config.dataloader.version = 1
            with open(config_file, "w") as f:
                f.write(config.to_json())
            trainer._after_step()
            self.assertEqual(dataloader.batch_sampler.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
