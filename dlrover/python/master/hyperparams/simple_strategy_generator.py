# Copyright 2022 The DLRover Authors. All rights reserved.
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

import threading
from typing import Dict, List

from dlrover.python.common.constants import NodeType
from dlrover.python.common.grpc import (
    DataLoaderConfig,
    OptimizerConfig,
    ParallelConfig,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.hyperparams.strategy_generator import (
    StrategyGenerator,
)
from dlrover.python.master.stats.reporter import JobMeta, LocalStatsReporter

gpu_stats = [
    {
        "index": 0,
        "total_memory_gb": 24,
        "used_memory_gb": 0,
    },
    {
        "index": 1,
        "total_memory_gb": 24,
        "used_memory_gb": 2.631,
    },
]

# Define model configuration as a dictionary
mock_model_config = {
    "block_size": 128,
    "n_layer": 6,
    "n_heads": 6,
    "n_embd": 384,
}

mock_dataloader_config = DataLoaderConfig(0, "simple_dataloader", 32, 2, 0)


class SimpleStrategyGenerator(StrategyGenerator):
    """
    A simple strategy generator for local optimization.
    This class is responsible for generating optimal configurations for data
    loaders and optimizers
    """

    def __init__(self, job_uuid):
        self._job_uuid = job_uuid
        self._stats_collector = LocalStatsReporter(JobMeta(job_uuid))

    def start(self):
        try:
            logger.info("Start local strategy generator")
            threading.Thread(
                target=self._extract_node_resource,
                name="strategy-generator",
                daemon=True,
            ).start()
        except Exception as e:
            logger.error(e)

    def generate_opt_strategy(
        self,
        gpu_stats=[],
        model_config=mock_model_config,
    ) -> ParallelConfig:
        node_samples = self._extract_node_resource()
        paral_configs: Dict[str, ParallelConfig] = {}
        for nodes in node_samples[NodeType.WORKER]:
            for node in nodes:
                gpu_stats = node.used_resource.gpu_stats
                paral_config = node.paral_config
                data_loader_config, optimizer_config = (
                    self._generate_dataloader_config(
                        gpu_stats, model_config, paral_config.dataloader_config
                    ),
                    self._generate_optimizer_config(),
                )
                paral_configs[node.name] = ParallelConfig(
                    data_loader_config, optimizer_config
                )
        logger.info("paral configs: %s", paral_configs["simple_node"])
        return paral_configs["simple_node"]

    def _generate_dataloader_config(
        self, gpu_stats, model_config, dataloader_config
    ):
        if gpu_stats == []:
            return DataLoaderConfig(0, "", 0, 0, 0)
        # Calculate the minimum remaining memory among GPUs
        min_remain_memory = min(
            entry["total_memory_gb"] - entry["used_memory_gb"]
            for entry in gpu_stats
        )

        # Update dataloader configuration version
        updated_version = dataloader_config.version + 1
        # Extract dataloader configuration values
        batch_size = dataloader_config.batch_size
        logger.info("batch size: %d", batch_size)

        # Extract model configuration values
        block_size = model_config["block_size"]
        n_layer = model_config["n_layer"]
        n_heads = model_config["n_heads"]
        n_embd = model_config["n_embd"]

        # Calculate the memory required for intermediate activation
        activation_memory_gb = (
            (
                34 * batch_size * block_size * n_embd
                + 5 * batch_size * (block_size**2) * n_heads
            )
            * n_layer
            / (1024 * 1024 * 1024)
        )

        # Calculate the updated batch size based on available memory
        updated_batch_size = int(
            batch_size + batch_size * min_remain_memory / activation_memory_gb
        )

        return DataLoaderConfig(
            updated_version,
            dataloader_config.dataloader_name,
            updated_batch_size,
            0,
            0,
        )

    def _generate_optimizer_config(self):
        return OptimizerConfig(5, 6)

    def _extract_node_resource(self) -> Dict[str, List[List[Node]]]:
        stats = self._stats_collector.get_runtime_stats()
        node_used_resources: Dict[str, List[List[Node]]] = {}
        node_used_resources[NodeType.WORKER] = []
        simple_node = Node(node_type="worker", node_id=0)
        simple_node.used_resource.gpu_stats = gpu_stats
        simple_node.paral_config.dataloader_config = mock_dataloader_config
        simple_node.name = "simple_node"
        node_used_resources[NodeType.WORKER].append([simple_node])
        if len(stats) == 0:
            logger.info("stats length is 0")
            return node_used_resources
        for node in stats[-1].running_nodes:
            logger.info("node type: %s", node.type)
            return node_used_resources
