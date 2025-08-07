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
from dlrover.python.unified.common.workload_base import ActorInfo, WorkerStage
from dlrover.python.unified.common.workload_desc import ElasticWorkloadDesc
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.actor_proxy import (
    invoke_actor_t,
    invoke_actors_t,
)


class ElasticManager:
    def __init__(self, workers: List[ActorInfo]):
        assert workers[0].spec.backend == "elastic"
        self.spec: ElasticWorkloadDesc = workers[0].spec
        self.workers: List[ActorInfo] = workers
        self.finished = False

        # self.perf_monitor = PerfMonitor()
        # self.diagnosis = DiagnosisMaster()
        self.node_check_manager = NodeCheckManager()
        self.stage = WorkerStage.READY

    async def check_workers(self, retry_count: int = 3):
        if not self.spec.comm_pre_check:
            logger.info(
                "Communication pre-check is disabled, skipping node-check."
            )
            return
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
                logger.info("Ask PrimeMaster to restart the nodes.")
                await invoke_actor_t(
                    PrimeMasterApi.restart_actors,
                    PrimeMasterApi.ACTOR_NAME,
                    actors=[node.name for node in abnormal_nodes],
                )
                logger.info("Restarted nodes, retrying node-check...")
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
        """Start the elastic training job."""
        if self.stage != WorkerStage.READY:
            raise Exception(
                f"Cannot start ElasticManager in stage {self.stage}, expected READY."
            )

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
        self._task = asyncio.create_task(self._monitor(), name="monitor_nodes")
        self.stage = WorkerStage.RUNNING
        logger.info("Elastic job started, monitoring nodes...")

    async def setup_workloads(self):
        logger.info("Start setup all elastic workloads...")
        start = time.time()

        await self.node_check_manager._setup_rendezvous_group(self.workers)
        logger.info("Setup torch process group for all nodes.")

        elapsed = time.time() - start
        logger.info(
            f"Finish setup all workloads, cost: {elapsed / 1000:.2f}ms"
        )

    async def _monitor(self):
        logger.info("Start monitoring nodes status...")
        while self.stage == WorkerStage.RUNNING:
            try:
                res = await invoke_actors_t(
                    remote_call.status, [node.name for node in self.workers]
                )
                logger.debug(f"Node status results: {res.results}")
                if all(it.is_terminal() for it in res.results):
                    logger.info("All nodes are finished.")
                    break
            except Exception as e:
                logger.warning(e)
            await asyncio.sleep(5)
        res = await invoke_actors_t(
            remote_call.destroy_torch_process_group,
            [node.name for node in self.workers],
        )
        res.log_errors()
        self.stage = WorkerStage.FINISHED

    async def handle_worker_restarted(self, worker_name: str):
        """Handle worker restart event."""
        worker = next((w for w in self.workers if w.name == worker_name), None)
        if not worker:
            logger.error(f"Worker {worker_name} not found.")
            return
        logger.info(f"Worker {worker_name} has been restarted.")
        if self.stage != WorkerStage.RUNNING:
            logger.info(
                f"Current stage is {self.stage}, skipping failover handling."
            )
            return
        asyncio.create_task(self.restart_job())
        # The stage is synchronously set to READY by restart_job，
        # so it's safe to invoke handle_worker_restarted multiple times.
        assert self.stage == WorkerStage.READY

    async def restart_job(self):
        """Restart the elastic job due to worker restart."""
        assert self.stage == WorkerStage.RUNNING, (
            f"Cannot restart job in stage {self.stage}, expected RUNNING."
        )
        logger.info("Restarting the elastic job due to worker restart.")
        self._task.cancel()
        self.stage = WorkerStage.READY  # Reset stage to READY for restart
        try:
            await self._task
        except asyncio.CancelledError:
            logger.info("Monitor task cancelled, proceeding with restart.")

        logger.info("Restarting all workers...")
        await invoke_actor_t(
            PrimeMasterApi.restart_actors,
            PrimeMasterApi.ACTOR_NAME,
            actors=[worker.name for worker in self.workers],
        )
        logger.info("Restarted workers, re-checking their status.")
        await self.check_workers()
        await self.start()
        logger.info("Restarted elastic job successfully.")
