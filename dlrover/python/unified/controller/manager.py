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
from threading import Thread
from typing import Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_base import MasterStage
from dlrover.python.unified.util.actor_helper import (
    invoke_actors_async,
    kill_actors,
)

from .config import JobConfig
from .schedule.graph import DLExecutionGraph
from .schedule.scheduler import Scheduler


class PrimeManager:
    INSTANCE: "PrimeManager"

    def __init__(self, config: JobConfig) -> None:
        self.config = config

        # Create all components
        self.graph = DLExecutionGraph.create(config.dl_config)
        self.scheduler: Scheduler = Scheduler(config)
        self.thread: Optional[Thread] = None

        # Runtime state
        self.stage: MasterStage = MasterStage.INIT
        logger.info(f"PrimeManager initialized with config: {config}")

        self.INSTANCE = self  # Singleton instance

    async def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when failed."""
        self.scheduler.allocate_placement_group(self.graph)
        await self.scheduler.create_actors(self.graph)
        logger.info("Finished creating actors for the job.")

        await self._nodes_check()

    async def _nodes_check(self):
        # check all nodes itself
        nodes = [node.name for node in self.graph.vertices]
        res = await invoke_actors_async(nodes, "self_check")
        res.raise_for_errors()
        logger.info("All nodes self-checked successfully.")

        # let masters pre-check nodes
        masters = [
            role.sub_master.name
            for role in self.graph.roles.values()
            if role.sub_master is not None
        ]
        res = await invoke_actors_async(masters, "check_workers")
        res.raise_for_errors()
        logger.info("Masters checked all workers successfully.")

    async def start(self):
        """Execute the job. Start tracking the job status."""
        self.stage = MasterStage.RUNNING
        self.save()
        nodes = [node.name for node in self.graph.vertices]
        res = await invoke_actors_async(nodes, "start")  # start all nodes
        res.raise_for_errors()

        res = await invoke_actors_async(nodes, "status")
        if any(it != "RUNNING" for it in res.results):
            statusMap = {
                node: node_status
                for node, node_status in zip(nodes, res.results)
            }
            raise Exception(f"Some nodes failed to start the job. {statusMap}")

        logger.info("Job started successfully.")
        self._task = asyncio.create_task(
            self._monitor_actors(), name="job_monitor"
        )

    async def _monitor_actors(self):
        """Monitor the nodes' status."""
        while self.stage == MasterStage.RUNNING:
            await asyncio.sleep(5)
            res = await invoke_actors_async(
                [node.name for node in self.graph.vertices], "status"
            )
            if all(it in ["FAILED", "FINISHED"] for it in res.results):
                if all(it == "FINISHED" for it in res.results):
                    logger.info("All nodes finished successfully.")
                else:
                    logger.info("All nodes finished, but some nodes failed.")
                break
        await self.stop()

    async def stop(self):
        """Stop the job execution. And clean up resources."""
        if (
            self.stage == MasterStage.STOPPING
            or self.stage == MasterStage.STOPPED
        ):
            return
        logger.info("Stopping the job...")
        self.stage = MasterStage.STOPPING
        kill_actors([node.name for node in self.graph.vertices])
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
        logger.info("Job stopped successfully.")
        self.stage = MasterStage.STOPPED
        self.save()

    def save(self):
        """Save the job state to persistent storage."""
        # This is a placeholder for saving the job state.
        # In a real implementation, this would save to a database or file.
        print(
            f"Job state saved: {self.stage}, "
            f"nodes: {[node.name for node in self.graph.vertices]}"
        )
        # TODO implement actual save logic. (Ref state_backend.py)
