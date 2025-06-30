import ray

from dlrover.python.unified.common.workload_defines import (
    MASTER_ACTOR_ID,
    ActorInfo,
)

from .api import PrimeMasterApi, PrimeMasterRemote
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

    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""
        actor = self.manager.graph.by_name.get(name)
        if actor is None:
            raise ValueError(f"Actor {name} not found.")
        return actor.to_actor_info()

    def get_actors_by_role(self, role: str) -> list[ActorInfo]:
        """Get all actors by role."""
        role_info = self.manager.graph.roles.get(role)
        if role_info is None:
            raise ValueError(f"Role {role} not found.")
        return [node.to_actor_info() for node in role_info.instances]

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
        return PrimeMasterApi

    # endregion
