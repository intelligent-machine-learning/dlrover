from typing import Literal, Protocol
from typing_extensions import TypeAlias

MasterStage: TypeAlias = Literal["INIT", "RUNNING", "STOPPING", "STOPPED"]


class ActorBase(Protocol):
    def status(self):
        """Get the state of the actor/node."""
        pass

    def self_check(self):
        """Check the actor/node itself."""
        pass

    def start(self):
        """Start the actor/node.If already started, do nothing."""


class Worker(ActorBase):
    pass


class SubMaster(Protocol):
    def check_workers(self):
        """Check the workers of the master."""
        ...
