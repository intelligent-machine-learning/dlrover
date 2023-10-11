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

import math
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

# TODO This is a mock model card configuration. We need to replace it with real
# model card configuration from model config reporter
mock_model_config = {
    "block_size": 128,
    "n_layer": 20,
    "n_heads": 20,
    "n_embd": 1280,
}


class SimpleStrategyGenerator(StrategyGenerator):
    """
    A simple strategy generator for local optimization.
    This class is responsible for generating optimal configurations for data
    loaders and optimizers
    """

    def __init__(self, job_uuid):
        self._job_uuid = job_uuid
        self._stats_collector = LocalStatsReporter(JobMeta(job_uuid))

    def generate_opt_strategy(
        self,
        gpu_stats=[],
        model_config=mock_model_config,
    ):
        node_samples = self._extract_node_resource()
        paral_configs: Dict[str, ParallelConfig] = {}
        for nodes in node_samples[NodeType.WORKER]:
            for node in nodes:
                gpu_stats = node.used_resource.gpu_stats
                paral_config = node.paral_config
                dataloader = self._generate_dataloader_config(
                    gpu_stats,
                    model_config,
                    paral_config.dataloader,
                )
                optimizer = self._generate_optimizer_config(
                    dataloader,
                    paral_config.optimizer,
                )
                paral_configs[node.id] = ParallelConfig(dataloader, optimizer)
                node.paral_config = paral_configs[node.id]
        if not paral_configs:
            logger.debug("No parallel config.")
            return None
        else:
            logger.debug(f"paral_configs: {paral_configs}")
            return paral_configs[0]

    def _generate_dataloader_config(
        self, gpu_stats, model_config, dataloader_config
    ):
        if gpu_stats == []:
            return dataloader_config
        # Calculate the minimum remaining memory among GPUs
        min_remain_memory = min(
            entry.total_memory_mb - entry.used_memory_mb for entry in gpu_stats
        )
        # To avoid the case that the remaining memory is too small and crashes
        # into OOM We set the minimum remaining memory to greater than 2400 MB
        # TODO: We need to replace 2400 with a more reasonable value
        if min_remain_memory > 2400:
            # Update dataloader configuration version
            updated_version = dataloader_config.version + 1
            # Extract dataloader configuration values
            batch_size = dataloader_config.batch_size

            # Extract model configuration values
            block_size = model_config["block_size"]
            n_layer = model_config["n_layer"]
            n_heads = model_config["n_heads"]
            n_embd = model_config["n_embd"]

            # Calculate the memory required for intermediate activation
            activation_memory_mb = (
                (
                    34 * batch_size * block_size * n_embd
                    + 5 * batch_size * (block_size**2) * n_heads
                )
                * n_layer
                / (1024**2)
            )
            try:
                updated_batch_size = int(
                    batch_size
                    + batch_size * min_remain_memory / activation_memory_mb
                )
            except ZeroDivisionError:
                updated_batch_size = batch_size

            logger.info(f"updated_batch_size: {updated_batch_size}")
            return DataLoaderConfig(
                updated_version,
                dataloader_config.dataloader_name,
                batch_size,
                updated_batch_size,
                0,
                0,
            )

    def _generate_optimizer_config(self, dataloader_config, optimizer_config):
        batch_size = dataloader_config.batch_size
        last_batch_size = dataloader_config.last_batch_size

        # Calculate the ratio between the latest batch_size
        # and the previous batch_size
        try:
            ratio = batch_size / last_batch_size
        except ZeroDivisionError:
            ratio = 1
        coefficient = math.sqrt(ratio)

        update_version = optimizer_config.version + 1

        # When the batch size is increased by a factor of ratio
        # increase the learning rate by a factor of sqrt(ratio)
        update_learning_rate = optimizer_config.learning_rate * coefficient

        # When the learning_rate is very small
        # update_weight_decay approximates to weight_decay * sqrt(ratio)
        # In order to mitigate the absolute error of the original formula
        # we set update_weight_decay = weight_decay * sqrt(ratio)
        update_weight_decay = optimizer_config.weight_decay * coefficient

        logger.info(
            f"Update optimizer with learning rate {update_learning_rate} "
            f"and weight decay {update_weight_decay}"
        )
        return OptimizerConfig(
            update_version,
            optimizer_config.optimizer_name,
            update_learning_rate,
            update_weight_decay,
        )

    def _extract_node_resource(self) -> Dict[str, List[List[Node]]]:
        stats = self._stats_collector.get_runtime_stats()
        node_used_resources: Dict[str, List[List[Node]]] = {}
        node_used_resources[NodeType.WORKER] = []
        if len(stats) == 0:
            logger.debug("There is no any training stats.")
            return node_used_resources
        else:
            for node in stats[-1].running_nodes:
                node_used_resources[NodeType.WORKER].append([node])
                logger.debug(f"node_used_resources: {node_used_resources}")
            return node_used_resources
