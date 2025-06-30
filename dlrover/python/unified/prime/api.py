from dlrover.python.unified.common.workload_defines import (
    MASTER_ACTOR_ID,
    ActorInfo,
    MasterStage,
)
from dlrover.python.unified.util.actor_helper import ActorProxy


class PrimeMasterRemote:
    """Stub for Remote interface for PrimeMaster."""

    def status(self) -> MasterStage:
        """Get the status of the master."""
        ...

    def start(self) -> None:
        """Start the master."""
        ...

    def stop(self) -> None:
        """Stop the master."""
        ...

    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""
        ...

    def get_workers_by_role(self, role: str) -> list[ActorInfo]:
        """Get all actors by role."""
        ...


PrimeMasterApi = ActorProxy.wrap(
    MASTER_ACTOR_ID, cls=PrimeMasterRemote, lazy=True
)
