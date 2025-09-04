from typing import List

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.util.extension_util import Extensible


class Extension(Extensible):
    INSTANCE: "Extension"

    @staticmethod
    def singleton() -> "Extension":
        if not Extension.INSTANCE:
            Extension.INSTANCE = Extension.build_mixed_class()()
        return Extension.INSTANCE

    async def relaunch_nodes_impl(self, nodes: List[NodeInfo]):
        """Relaunch the specified nodes.
        @param nodes: The list of ray node IDs to relaunch.
        """
        raise NotImplementedError("Relaunch is not implemented")
