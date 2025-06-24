from threading import Thread
import time
from typing import Dict, Optional

from dlrover.python.hybrid.config import JobConfig
from dlrover.python.hybrid.defines import MasterStage
from dlrover.python.hybrid.schedule.graph import DLExecutionGraph
from dlrover.python.hybrid.schedule.scheduler import Node, Scheduler
from dlrover.python.common.log import default_logger as logger


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
        self.placement = Placement()  # Placeholder for actual placement strategy
        self.scheduler: Scheduler = Scheduler(config.scheduling_strategy_type)
        self.thread: Optional[Thread] = None

        # Runtime state
        self.stage: MasterStage = "INIT"
        self.nodes: Dict[str, Node] = {}  # name -> node mapping
        logger.info(f"HybridManager initialized with config: {config}")

    def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when fail."""
        self.placement.allocate_placement_group(self.graph)
        self.nodes = {
            node.name: node for node in self.graph.prepare_nodes()
        }  # include workers and subMasters

        self.scheduler.create_nodes(self.nodes.values())  # create actors for all nodes
        logger.info("Finished creating nodes for the job.")

        self._nodes_check()

    def _nodes_check(self):
        self.scheduler.execute(
            self.nodes.values(), "self_check"
        )  # check all nodes itself
        logger.info("All nodes self-checked successfully.")

        masters = [node for node in self.nodes.values() if node.kind == "master"]
        self.scheduler.execute(masters, "check_workers")  # let masters pre-check nodes
        logger.info("Masters checked all workers successfully.")

    def start(self):
        """Execute the job. Start tracking the job status."""
        self.stage = "RUNNING"
        self.save()
        self.scheduler.execute(self.nodes.values(), "start")  # start all nodes
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
        self.scheduler.cleanup(self.nodes.values())
        if self.thread is not None:
            self.thread.join()
        logger.info("Job stopped successfully.")
        self.stage = "STOPPED"
        self.save()

    def save(self):
        """Save the job state to persistent storage."""
        # This is a placeholder for saving the job state.
        # In a real implementation, this would save to a database or file.
        print(f"Job state saved: {self.stage}, nodes: {list(self.nodes.keys())}")
