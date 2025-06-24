from dataclasses import dataclass
from typing import Literal, Protocol

from typing_extensions import TypeAlias

MASTER_ACTOR_ID = "__hybrid_master__"
MasterStage: TypeAlias = Literal["INIT", "RUNNING", "STOPPING", "STOPPED"]


@dataclass
class NodeInfo:
    """Information about a node. Exposed to workers and sub-masters."""

    name: str
    role: str
    config: dict

    rank: int = 0
    local_rank: int = 0


class ActorBase(Protocol):
    def status(self):
        """Get the state of the actor/node."""
        pass

    def self_check(self):
        """Check the actor/node itself."""
        pass

    def start(self):
        """Start the actor/node.If already started, do nothing."""

    # for sub-master
    def check_workers(self):
        """Check the workers of the master."""
        ...
