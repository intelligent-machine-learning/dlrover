# Copyright 2025 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Dict, Set

from ray.actor import ActorClass
from ray.util.scheduling_strategies import SchedulingStrategyT

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.enums import SchedulingStrategyType
from dlrover.python.unified.common.workload_config import ResourceDesc
from dlrover.python.unified.common.workload_defines import JobInfo
from dlrover.python.unified.prime.schedule.graph import (
    DLExecutionGraph,
    PlacementGroupSpec,
)
from dlrover.python.unified.util.actor_helper import (
    BatchInvokeResult,
    invoke_actors_async,
)


@dataclass
class RayNodeSpec:
    name: str
    cls: ActorClass
    options: Dict[str, Any]  # as kwargs for actor
    resource: ResourceDesc = field(default_factory=ResourceDesc)
    envs: Dict[str, str] = field(default_factory=dict)
    scheduling_strategy: SchedulingStrategyT = None


class Placement:
    def allocate_placement_group(self, graph: DLExecutionGraph):
        """Allocate placement group based on the execution graph."""
        # update vertices with placement group info
        ...
        # TODO implement the logic to allocate placement groups
        #  Refer: placement.py
        #  Result: set PlacementGroupSpec and bundle_index for vertices


class Scheduler:
    def __init__(self, strategy: SchedulingStrategyType) -> None:
        self.strategy = strategy

        # Placeholder for actual placement strategy
        self.placement = Placement()

        self.__pg = None  # Placement group for actors

    def create_pgs(self, pgs: Set[PlacementGroupSpec]):
        """Create placement groups for
        the given set of placement group specs."""
        # TODO implement the logic to create placement groups

    async def create_nodes(self, graph: DLExecutionGraph, job_info: JobInfo):
        """Create/Get actors for all nodes in the execution graph."""
        # 0. create placement group if not exists
        self.placement.allocate_placement_group(graph)
        """Create placement group if not exists."""
        pgs = set(
            it.placement_group
            for it in graph.vertices
            if it.placement_group is not None
        )
        self.create_pgs(pgs)

        # 1. ray create_or_exists actors
        for role in graph.roles.values():
            for worker in role.instances:
                spec = RayNodeSpec(
                    name=worker.name,
                    resource=role.spec.instance_resource,
                    cls=role.spec.get_worker_cls(),  # type: ignore[assignment]
                    envs=worker.get_envs(),
                    scheduling_strategy=None,  # no scheduling strategy for now
                    options={
                        "job_info": job_info,
                        "actor_info": worker.to_actor_info(),
                    },
                )
                self.create_node(spec)
            if role.sub_master is not None:
                # Create sub-master node if it exists
                spec = RayNodeSpec(
                    name=role.sub_master.name,
                    # resource=role.spec.instance_resource,
                    cls=role.spec.get_master_cls(),  # type: ignore[assignment]
                    envs=role.sub_master.get_envs(),
                    scheduling_strategy=None,  # no scheduling strategy for now
                    options={
                        "job_info": job_info,
                        "actor_info": role.sub_master.to_actor_info(),
                    },
                )
                self.create_node(spec)
        logger.info("Finished creating nodes for the job.")

        # 2. Check actors with ping
        res: BatchInvokeResult[str] = await invoke_actors_async(
            [node.name for node in graph.vertices], "status"
        )
        logger.info(f"Actors status: {res.as_dict()}")

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
