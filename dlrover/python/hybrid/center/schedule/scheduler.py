from dataclasses import dataclass
from typing import Any, Dict

from ray.actor import ActorClass
from ray.util.scheduling_strategies import SchedulingStrategyT

from dlrover.python.common.log import default_logger as logger
from dlrover.python.hybrid.center.config import ResourceDesc
from dlrover.python.hybrid.center.schedule.graph import DLExecutionGraph
from dlrover.python.hybrid.util.actor_helper import (
    invoke_actors,
)
from dlrover.python.hybrid.worker.worker import Worker
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.enums import SchedulingStrategyType


@dataclass
class RayNodeSpec:
    name: str
    resource: ResourceDesc
    envs: Dict[str, str]
    cls: ActorClass
    options: Dict[str, Any]  # as kwargs for actor
    scheduling_strategy: SchedulingStrategyT = None


class Placement:
    def allocate_placement_group(self, graph: DLExecutionGraph):
        """Allocate placement group based on the execution graph."""
        # update vertices with placement group info
        pass


class Scheduler:
    def __init__(self, strategy: SchedulingStrategyType) -> None:
        self.strategy = strategy

        # Placeholder for actual placement strategy
        self.placement = Placement()

        self.__pg = None  # Placement group for actors

    def create_pgs(self, pgs):
        pass

    def create_nodes(self, graph: DLExecutionGraph):
        """Create/Get actors for all nodes in the execution graph."""
        # 0. create placement group if not exists
        """TODO: Create placement group if not exists."""

        # 1. ray create_or_exists actors
        for vertex in graph.vertices:
            spec = RayNodeSpec(
                name=vertex.name,
                resource=vertex.spec.instance_resource,
                envs=vertex.get_envs(),
                cls=Worker,  # type: ignore[assignment]
                scheduling_strategy=None,  # no scheduling strategy for now
                options={},  # options is used for actor's kwargs
            )
            self.create_node(spec)
        logger.info("Finished creating nodes for the job.")

        # 2. Check actors with ping
        result = invoke_actors(
            [node.name for node in graph.vertices], "status"
        )
        logger.info(f"Actors status: {result}")

    def create_node(self, node: RayNodeSpec):
        runtime_env: dict = {
            "env_vars": node.envs,
        }
        # setup working dir
        if DLWorkloadEnv.WORKING_DIR in node.envs:
            runtime_env["working_dir"] = node.envs[DLWorkloadEnv.WORKING_DIR]

        node.cls.options(
            name=node.name,
            lifetime="detached",
            max_restarts=-1,  # Allow unlimited restarts
            get_if_exists=True,
            num_cpus=node.resource.cpu,
            memory=node.resource.memory,
            num_gpus=node.resource.gpu,
            resources=node.resource.user_defined,
            runtime_env=runtime_env,
            scheduling_strategy=node.scheduling_strategy,
        ).remote(**node.options)
        logger.info(f"Created {node}")
