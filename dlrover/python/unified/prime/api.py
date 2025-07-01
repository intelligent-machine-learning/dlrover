from abc import abstractmethod
from typing import Protocol

from dlrover.python.unified.common.workload_defines import (
    MASTER_ACTOR_ID,
    ActorInfo,
    MasterStage,
)
from dlrover.python.unified.util.actor_helper import ActorProxy


class PrimeMasterRemote(Protocol):
    """Stub for Remote interface for PrimeMaster."""

    @abstractmethod
    def status(self) -> MasterStage:
        """Get the status of the master."""

    @abstractmethod
    def start(self) -> None:
        """Start the master."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the master."""

    @abstractmethod
    def shutdown(self) -> None:
        """Force shutdown the master."""

    @abstractmethod
    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""

    @abstractmethod
    def get_workers_by_role(self, role: str) -> list[ActorInfo]:
        """Get all actors by role."""


PrimeMasterApi = ActorProxy.wrap(
    MASTER_ACTOR_ID, cls=PrimeMasterRemote, lazy=True
)
