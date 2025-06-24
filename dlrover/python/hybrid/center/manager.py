import time
from threading import Thread
from typing import Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.hybrid.center.config import JobConfig
from dlrover.python.hybrid.defines import MasterStage
from dlrover.python.hybrid.center.schedule.graph import DLExecutionGraph
from dlrover.python.hybrid.center.schedule.scheduler import Scheduler
from dlrover.python.hybrid.util.actor_helper import invoke_actors, kill_actors


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

    def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when fail."""
        self.placement.allocate_placement_group(self.graph)
        self.scheduler.create_nodes(self.graph)  # create actors for all nodes
        logger.info("Finished creating nodes for the job.")

        self._nodes_check()

    def _nodes_check(self):
        # check all nodes itself
        nodes = [node.name for node in self.graph.vertices]
        invoke_actors(nodes, "self_check")
        logger.info("All nodes self-checked successfully.")

        # let masters pre-check nodes
        invoke_actors(nodes, "check_workers")
        logger.info("Masters checked all workers successfully.")

    def start(self):
        """Execute the job. Start tracking the job status."""
        self.stage = "RUNNING"
        self.save()
        nodes = [node.name for node in self.graph.vertices]
        invoke_actors(nodes, "start")  # start all nodes
        logger.info("Job started successfully.")
        self.thread = Thread(
            target=self._monitor_nodes, name="job_monitor", daemon=True
        )
        self.thread.start()

    def _monitor_nodes(self):
        """Monitor the nodes status."""
        while self.stage == "RUNNING":
            pass
            time.sleep(5)
        self.stop()

    def stop(self):
        """Stop the job execution. And clean up resources."""
        if self.stage == "STOPPING" or self.stage == "STOPPED":
            return
        logger.info("Stopping the job...")
        self.stage = "STOPPING"
        kill_actors([node.name for node in self.graph.vertices])
        if self.thread is not None:
            self.thread.join()
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
