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
import unittest
from typing import Dict, List
from unittest.mock import patch

from dlrover.python.common.constants import NodeType
from dlrover.python.common.grpc import (
    DataLoaderConfig,
    GPUStats,
    OptimizerConfig,
)
from dlrover.python.common.node import Node
from dlrover.python.master.hyperparams.simple_strategy_generator import (
    SimpleStrategyGenerator,
)


class TestLocalStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.job_uuid = "1234"
        self._strategy_generator = SimpleStrategyGenerator(self.job_uuid)

    def test_generate_opt_strategy(self):
        gpu_stats = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            ),
            GPUStats(
                index=1,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            ),
        ]

        model_config = {
            "block_size": 128,
            "n_layer": 20,
            "n_heads": 20,
            "n_embd": 1280,
        }
        dataloader_config = DataLoaderConfig(
            0, "simple_dataloader", 0, 32, 2, 0
        )
        optimizer_config = OptimizerConfig(1, "SGD", 0.01, 0.001)
        node_used_resources: Dict[str, List[List[Node]]] = {}
        node_used_resources[NodeType.WORKER] = []
        simple_node = Node(node_type="worker", node_id=0)
        simple_node.used_resource.gpu_stats = gpu_stats
        simple_node.paral_config.dataloader = dataloader_config
        simple_node.paral_config.optimizer = optimizer_config
        simple_node.name = "simple_node"
        node_used_resources[NodeType.WORKER].append([simple_node])
        with patch.object(
            SimpleStrategyGenerator, "_extract_node_resource"
        ) as mock_extract_node_resource:
            mock_extract_node_resource.return_value = node_used_resources
            expected_dataloader_config = DataLoaderConfig(
                1, "simple_dataloader", 32, 177, 0, 0
            )
            expected_optimizer_config = OptimizerConfig(
                2,
                "SGD",
                0.01 * math.sqrt(177 / 32),
                0.001 * math.sqrt(177 / 32),
            )

            result = self._strategy_generator.generate_opt_strategy(
                gpu_stats, model_config
            )

            self.assertEqual(expected_dataloader_config, result.dataloader)
            self.assertEqual(expected_optimizer_config, result.optimizer)


if __name__ == "__main__":
    unittest.main()
