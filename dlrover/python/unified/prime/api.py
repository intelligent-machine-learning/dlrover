from typing import TYPE_CHECKING, List

from dlrover.python.unified.common.node_defines import (
    MASTER_ACTOR_ID,
    NodeInfo,
)
from dlrover.python.unified.util.actor_helper import ActorProxy

if TYPE_CHECKING:
    from .master import HybridMaster


def _proxy() -> "HybridMaster":
    return ActorProxy.wrap(MASTER_ACTOR_ID)


class PrimeMasterApi:
    @staticmethod
    def get_nodes_by_role(role: str) -> List[NodeInfo]:
        return _proxy().get_nodes_by_role(role)
