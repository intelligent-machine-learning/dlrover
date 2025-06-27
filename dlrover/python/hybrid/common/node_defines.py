from dataclasses import dataclass
from enum import Enum
from typing import Literal

from typing_extensions import TypeAlias

from dlrover.python.hybrid.common.workload_config import WorkloadDesc

MASTER_ACTOR_ID = "__hybrid_master__"
MasterStage: TypeAlias = Literal["INIT", "RUNNING", "STOPPING", "STOPPED"]


class WorkerStage(str, Enum):
    """Stages of a worker actor."""

    INIT = "INIT"
    PENDING = "PENDING"  # Checking
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"

    def is_terminal(self) -> bool:
        """Check if the stage is terminal."""
        return self in {WorkerStage.FINISHED, WorkerStage.FAILED}


@dataclass
class NodeInfo:
    """Information about a node. Exposed to workers and sub-masters."""

    name: str
    role: str
    spec: WorkloadDesc

    rank: int = 0
    local_rank: int = 0


class ActorBase:
    def __init__(self, info: NodeInfo) -> None:
        """Initialize the actor with node information."""
        self.node_info = info
        self.stage: WorkerStage = WorkerStage.INIT
        self._setup()

    # Hook methods for subclasses to implement
    def _setup(self):
        """Setup the actor/node."""
        pass

    def status(self):
        """Get the state of the actor/node."""
        return self.stage

    def self_check(self):
        """Check the actor/node itself."""
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return  # already in the expected stage
        print("Worker self check")

    def start(self):
        """Start the actor/node.If already started, do nothing."""
        print("Worker started: No Implementation")

    # for sub-master
    def check_workers(self):
        """Check the workers of the master."""
        pass

    # Helper methods for subclasses to use

    def _update_stage_force(
        self, stage: WorkerStage, expected: WorkerStage = None
    ):
        """Update the stage of the actor/node."""
        if expected is not None and self.stage != expected:
            raise RuntimeError(
                f"Cannot update stage from {self.stage} to {stage}, expected {expected}."
            )
        self.stage = stage
        print(f"Actor {self.node_info.name} updated to stage: {self.stage}")

    def _update_stage_if(self, stage: WorkerStage, expected: WorkerStage):
        """Update the stage of the actor/node if the current stage matches the expected stage."""
        if self.stage != expected:
            print(
                f"Actor {self.node_info.name} is not in expected stage: {expected}, current stage: {self.stage}"
            )
            return False  # not in the expected stage
        self.stage = stage
        print(f"Actor {self.node_info.name} updated to stage: {self.stage}")
        return True
