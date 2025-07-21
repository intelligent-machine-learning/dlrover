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
from dlrover.python.unified.controller import remote_call
from dlrover.python.unified.util.actor_helper import (
    kill_actors,
)
from dlrover.python.unified.util.actor_proxy import invoke_actors_t

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
        self.exit_code: int = 0
        logger.info(f"PrimeManager initialized with config: {config}")
        self._stopped_event = asyncio.Event()

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
        res = await invoke_actors_t(remote_call.self_check, nodes)
        res.raise_for_errors()
        logger.info("All nodes self-checked successfully.")

        # let masters pre-check nodes
        masters = [
            role.sub_master.name
            for role in self.graph.roles.values()
            if role.sub_master is not None
        ]
        res = await invoke_actors_t(remote_call.check_workers, masters)
        res.raise_for_errors()
        logger.info("Masters checked all workers successfully.")

    async def start(self):
        """Execute the job. Start tracking the job status."""
        self.stage = MasterStage.RUNNING
        self.save()
        nodes = [node.name for node in self.graph.vertices]
        res = await invoke_actors_t(remote_call.start, nodes)
        res.raise_for_errors()

        res = await invoke_actors_t(remote_call.status, nodes)
        if any(it != "RUNNING" for it in res.results):
            statusMap = {
                node: node_status
                for node, node_status in zip(nodes, res.results)
            }
            raise Exception(f"Some nodes failed to start the job. {statusMap}")

        logger.info("Job started successfully.")
        asyncio.create_task(self._main_loop(), name="job_monitor")

    async def _main_loop(self):
        """Monitor the nodes' status."""
        while self.stage == MasterStage.RUNNING:
            await asyncio.sleep(5)
            res = await invoke_actors_t(
                remote_call.status, self.graph.vertices
            )
            if all(it in ["FAILED", "FINISHED"] for it in res.results):
                if all(it == "FINISHED" for it in res.results):
                    self.request_stop("All nodes finished successfully.")
                else:
                    self.exit_code = 1
                    self.request_stop(
                        "All nodes finished, but some nodes failed."
                    )
                break

        assert self.stage == MasterStage.STOPPING, (
            f"Job stage should be STOPPING, but got {self.stage}."
        )
        kill_actors([node.name for node in self.graph.vertices])
        logger.info("Job stopped successfully.")
        self.stage = MasterStage.STOPPED
        self.save()
        self._stopped_event.set()

    def request_stop(self, reason: str):
        """Stop the job execution. And clean up resources."""
        if (
            self.stage == MasterStage.STOPPING
            or self.stage == MasterStage.STOPPED
        ):
            return
        logger.info(f"Requesting to stop the job: {reason}")
        self.stage = MasterStage.STOPPING

    async def wait(self):
        """Wait for the job to finish."""
        await self._stopped_event.wait()
        assert self.stage == MasterStage.STOPPED

    def save(self):
        """Save the job state to persistent storage."""
        # This is a placeholder for saving the job state.
        # In a real implementation, this would save to a database or file.
        logger.info(
            f"Job state saved: {self.stage}, "
            f"nodes: {[node.name for node in self.graph.vertices]}"
        )
        # TODO implement actual save logic. (Ref state_backend.py)
