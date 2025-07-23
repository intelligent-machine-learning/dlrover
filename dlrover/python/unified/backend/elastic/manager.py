# Copyright 2025 The DLRover Authors. All rights reserved.
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

import asyncio
import time
from typing import List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.backend.elastic import remote_call
from dlrover.python.unified.backend.elastic.node_check_manager import (
    NodeCheckManager,
)
from dlrover.python.unified.common.workload_base import ActorInfo
from dlrover.python.unified.util.actor_helper import restart_actors
from dlrover.python.unified.util.actor_proxy import (
    invoke_actors_t,
)


class ElasticManager:
    def __init__(self, workers: List[ActorInfo]):
        self.workers: List[ActorInfo] = workers
        self.finished = False

        # self.perf_monitor = PerfMonitor()
        # self.diagnosis = DiagnosisMaster()
        self.node_check_manager = NodeCheckManager()

    def _prepare(self):
        pass

    async def _restart_nodes(self, nodes: List[ActorInfo]):
        """
        Restart the specified nodes.
        This is a placeholder for actual restart logic.
        """
        await restart_actors([node.name for node in nodes])
        res = await invoke_actors_t(
            remote_call.status, [node.name for node in nodes]
        )
        print(f"Restarted nodes status: {res.as_dict()}")

    async def check_workers(self, retry_count: int = 3):
        logger.info("Do node-check for all nodes...")
        delays = await self.node_check_manager.check_nodes(self.workers)
        abnormal_nodes = self.node_check_manager.find_abnormal_nodes(
            self.workers, delays, threshold=300.0
        )
        if abnormal_nodes:
            logger.warning(
                f"Node-check found {len(abnormal_nodes)} abnormal nodes: "
                f"{', '.join(str(node) for node in abnormal_nodes)}"
            )
            if retry_count > 0:
                await self._restart_nodes(abnormal_nodes)
                return await self.check_workers(retry_count - 1)
            raise Exception(
                "Node-check failed, some nodes are not ready to start the job."
            )
        straggling_nodes = self.node_check_manager.find_straggling_nodes(
            self.workers, delays
        )
        if straggling_nodes:
            logger.warning(
                f"Node-check found {len(straggling_nodes)} straggling nodes: "
                f"{', '.join(str(node) for node in straggling_nodes)}"
            )
            # No action taken for straggling nodes
        logger.info("Node-check finished for all nodes.")

    async def start(self):
        # Initialize the elastic client here
        logger.info("Start job execution.")
        await self.setup_workloads()
        res = await invoke_actors_t(
            remote_call.start_elastic_job, [node.name for node in self.workers]
        )
        res.raise_for_errors()
        res = await invoke_actors_t(
            remote_call.status, [node.name for node in self.workers]
        )
        if any(it != "RUNNING" for it in res.results):
            raise Exception("Some nodes failed to start the job.")
        asyncio.create_task(self._monitor(), name="monitor_nodes")

    async def setup_workloads(self):
        logger.info("Start setup all workloads...")
        start = time.time()

        await self.node_check_manager._setup_rendezvous_group(self.workers)
        logger.info("Setup torch process group for all nodes.")

        elapsed = time.time() - start
        logger.info(
            f"Finish setup all workloads, cost: {elapsed / 1000:.2f}ms"
        )

    async def _monitor(self):
        logger.info("Start monitoring nodes status...")
        while not self.finished:
            try:
                res = await invoke_actors_t(
                    remote_call.status, [node.name for node in self.workers]
                )
                logger.debug(f"Node status results: {res.results}")
                if all(it.is_terminal() for it in res.results):
                    self.finished = True
                    logger.info("All nodes are finished.")
                    break
            except Exception as e:
                logger.warning(e)
                await asyncio.sleep(30)
            await asyncio.sleep(5)
        res = await invoke_actors_t(
            remote_call.destroy_torch_process_group,
            [node.name for node in self.workers],
        )
        res.raise_for_errors()
