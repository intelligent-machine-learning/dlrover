from dlrover.python.unified.common.node_defines import (
    MASTER_ACTOR_ID,
    MasterStage,
    NodeInfo,
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

    def get_node(self, node_name: str) -> NodeInfo:
        """Get a node by name."""
        ...

    def get_nodes_by_role(self, role: str) -> list[NodeInfo]:
        """Get all nodes by role."""
        ...


PrimeMasterApi = ActorProxy.wrap(
    MASTER_ACTOR_ID, cls=PrimeMasterRemote, lazy=True
)
