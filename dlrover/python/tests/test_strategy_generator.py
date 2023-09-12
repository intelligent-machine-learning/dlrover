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

from dlrover.python.common.grpc import (
    DataLoaderConfig,
    OptimizerConfig,
    ParallelConfig,
)
from dlrover.python.master.hyperparams.simple_strategy_generator import (
    SimpleStrategyGenerator,
)


class TestLocalStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.job_uuid = "1234"
        self._strategy_generator = SimpleStrategyGenerator(self.job_uuid)

    def test_generate_opt_strategy(self):
        gpu_stats = [
            {
                "index": 0,
                "total_memory_gb": 40,
                "used_memory_gb": 2,
            },
            {
                "index": 1,
                "total_memory_gb": 40,
                "used_memory_gb": 12,
            },
        ]

        model_config = {
            "block_size": 128,
            "n_layer": 6,
            "n_heads": 6,
            "n_embd": 384,
        }

        dataloader_config = DataLoaderConfig(0, "simple_dataloader", 32, 2, 0)
        optimizer_config = OptimizerConfig(0, 0)
        paral_config = ParallelConfig(dataloader_config, optimizer_config)

        expected_dataloader_config = DataLoaderConfig(
            1, "simple_dataloader", 1800, 0, 0
        )
        expected_optimizer_config = OptimizerConfig(5, 6)

        result = self._strategy_generator.generate_opt_strategy(
            gpu_stats, model_config
        )
        print(result)
        self.assertEqual(expected_dataloader_config, result.dataloader)
        self.assertEqual(expected_optimizer_config, result.optimizer)


if __name__ == "__main__":
    unittest.main()
