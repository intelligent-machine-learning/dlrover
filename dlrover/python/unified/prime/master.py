import ray

from dlrover.python.unified.common.node_defines import (
    MASTER_ACTOR_ID,
    NodeInfo,
)
from dlrover.python.unified.util.actor_helper import ActorProxy

from .api import PrimeMasterRemote
from .config import JobConfig
from .manager import PrimeManager


class PrimeMaster:
    def __init__(self, config: JobConfig):
        assert ray.get_runtime_context().get_actor_name() == MASTER_ACTOR_ID, (
            f"PrimeMaster must be initialized as a Ray actor with the name '{MASTER_ACTOR_ID}'."
        )

        self.manager = PrimeManager(config)

    def status(self):
        return self.manager.stage

    async def start(self):
        await self.manager.prepare()
        await self.manager.start()

    async def stop(self):
        await self.manager.stop()

    # region RPC

    def get_node(self, node_name: str) -> NodeInfo:
        """Get a node by name."""
        node = self.manager.graph.by_name.get(node_name)
        if node is None:
            raise ValueError(f"Node {node_name} not found.")
        return node.to_node_info()

    def get_nodes_by_role(self, role: str) -> list[NodeInfo]:
        """Get all nodes by role."""
        role_info = self.manager.graph.roles.get(role)
        if role_info is None:
            raise ValueError(f"Role {role} not found.")
        return [node.to_node_info() for node in role_info.instances]

    @staticmethod
    def create(
        config: JobConfig, detached: bool = True
    ) -> "PrimeMasterRemote":
        """Create a PrimeMaster instance."""
        ray.remote(PrimeMaster).options(
            name=MASTER_ACTOR_ID,
            lifetime="detached" if detached else "normal",
            num_cpus=config.master_cpu,
            memory=config.master_mem,
            max_restarts=config.master_max_restart,
            max_concurrency=64,
        ).remote(config)
        return ActorProxy.wrap(MASTER_ACTOR_ID, PrimeMasterRemote)

    # endregion
