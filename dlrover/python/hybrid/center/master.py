import ray

from dlrover.python.hybrid.center.config import JobConfig
from dlrover.python.hybrid.center.manager import HybridManager
from dlrover.python.hybrid.defines import MASTER_ACTOR_ID, NodeInfo
from dlrover.python.hybrid.util.actor_helper import ActorProxy


class HybridMaster:
    def __init__(self, config: JobConfig):
        assert ray.get_runtime_context().get_actor_name() == MASTER_ACTOR_ID, (
            f"HybridMaster must be initialized as a Ray actor with the name '{MASTER_ACTOR_ID}'."
        )

        self.manager = HybridManager(config)

    def status(self):
        return self.manager.stage

    def start(self):
        self.manager.prepare()
        self.manager.start()

    def stop(self):
        self.manager.stop()

    # region RPC

    def get_node(self, node_name: str) -> NodeInfo:
        """Get a node by name."""
        node = self.manager.graph.by_name.get(node_name)
        if node is None:
            raise ValueError(f"Node {node_name} not found.")
        return node.to_node_info()

    def get_nodes_by_role(self, role: str) -> list[NodeInfo]:
        """Get all nodes by role."""
        nodes = self.manager.graph.by_role.get(role, [])
        return [node.to_node_info() for node in nodes]

    @staticmethod
    def create(config: JobConfig, detached: bool = True) -> "HybridMaster":
        """Create a HybridMaster instance."""
        ray.remote(HybridMaster).options(
            name=MASTER_ACTOR_ID,
            lifetime="detached" if detached else "normal",
            num_cpus=config.master_cpu,
            memory=config.master_mem,
            max_restarts=config.master_max_restart,
        ).remote(config)
        return ActorProxy.wrap(MASTER_ACTOR_ID)

    # endregion
