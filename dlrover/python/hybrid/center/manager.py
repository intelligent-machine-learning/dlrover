import asyncio
from threading import Thread
from typing import Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.hybrid.center.config import JobConfig
from dlrover.python.hybrid.center.schedule.graph import DLExecutionGraph
from dlrover.python.hybrid.center.schedule.scheduler import Scheduler
from dlrover.python.hybrid.common.node_defines import MasterStage
from dlrover.python.hybrid.util.actor_helper import (
    invoke_actors_async,
    kill_actors,
)


class Placement:
    def allocate_placement_group(self, graph: DLExecutionGraph):
        """Allocate placement group based on the execution graph."""
        # update vertices with placement group info
        pass


class HybridManager:
    def __init__(self, config: JobConfig) -> None:
        self.config = config

        # Create all components
        self.graph = DLExecutionGraph.create(config.dl_config)
        self.placement = (
            Placement()
        )  # Placeholder for actual placement strategy
        self.scheduler: Scheduler = Scheduler(config.scheduling_strategy_type)
        self.thread: Optional[Thread] = None

        # Runtime state
        self.stage: MasterStage = "INIT"
        logger.info(f"HybridManager initialized with config: {config}")

    async def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when fail."""
        self.placement.allocate_placement_group(self.graph)
        await self.scheduler.create_nodes(
            self.graph
        )  # create actors for all nodes
        logger.info("Finished creating nodes for the job.")

        await self._nodes_check()

    async def _nodes_check(self):
        # check all nodes itself
        nodes = [node.name for node in self.graph.vertices]
        await invoke_actors_async(nodes, "self_check")
        logger.info("All nodes self-checked successfully.")

        # let masters pre-check nodes
        masters = [
            role.sub_master.name
            for role in self.graph.roles.values()
            if role.sub_master is not None
        ]
        await invoke_actors_async(masters, "check_workers")
        logger.info("Masters checked all workers successfully.")

    async def start(self):
        """Execute the job. Start tracking the job status."""
        self.stage = "RUNNING"
        self.save()
        nodes = [node.name for node in self.graph.vertices]
        await invoke_actors_async(nodes, "start")  # start all nodes
        status = await invoke_actors_async(nodes, "status")
        if any(it != "RUNNING" for it in status):
            statusMap = {
                node.name: node_status
                for node, node_status in zip(self.graph.vertices, status)
            }
            raise Exception(f"Some nodes failed to start the job. {statusMap}")
        logger.info("Job started successfully.")
        self._task = asyncio.create_task(
            self._monitor_nodes(), name="job_monitor"
        )

    async def _monitor_nodes(self):
        """Monitor the nodes status."""
        while self.stage == "RUNNING":
            await asyncio.sleep(5)
            results = await invoke_actors_async(
                [node.name for node in self.graph.vertices], "status"
            )
            if all(it in ["FAILED", "FINISHED"] for it in results):
                logger.info("All nodes are finished or failed.")
                break
        await self.stop()

    async def stop(self):
        """Stop the job execution. And clean up resources."""
        if self.stage == "STOPPING" or self.stage == "STOPPED":
            return
        logger.info("Stopping the job...")
        self.stage = "STOPPING"
        kill_actors([node.name for node in self.graph.vertices])
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
        logger.info("Job stopped successfully.")
        self.stage = "STOPPED"
        self.save()

    def save(self):
        """Save the job state to persistent storage."""
        # This is a placeholder for saving the job state.
        # In a real implementation, this would save to a database or file.
        print(
            f"Job state saved: {self.stage}, nodes: {[node.name for node in self.graph.vertices]}"
        )
