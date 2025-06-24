from typing import TYPE_CHECKING, List

from dlrover.python.hybrid.defines import MASTER_ACTOR_ID, NodeInfo
from dlrover.python.hybrid.util.actor_helper import (
    ActorProxy,
)

if TYPE_CHECKING:
    from dlrover.python.hybrid.center.master import HybridMaster


def _proxy() -> "HybridMaster":
    return ActorProxy.wrap(MASTER_ACTOR_ID)


def get_nodes_by_role(role: str) -> List[NodeInfo]:
    return _proxy().get_nodes_by_role(role)
