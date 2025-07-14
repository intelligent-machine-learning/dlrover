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
from typing import Any, Dict, List, Optional

from ray.actor import ActorClass
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
    SchedulingStrategyT,
)

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.workload_config import ResourceDesc
from dlrover.python.unified.controller.config import (
    ACCELERATOR_TYPE,
    JobConfig,
)
from dlrover.python.unified.util.actor_helper import (
    BatchInvokeResult,
    invoke_actors_async,
)

from .graph import DLExecutionGraph


@dataclass
class RayActorSpec:
    """Specification for a Ray actor. private to scheduler."""

    name: str
    cls: ActorClass
    options: Dict[str, Any]  # as kwargs for actor
    resource: ResourceDesc = field(default_factory=ResourceDesc)
    envs: Dict[str, str] = field(default_factory=dict)
    scheduling_strategy: SchedulingStrategyT = None


class Scheduler:
    def __init__(self, config: JobConfig) -> None:
        self._config = config
        self._pg: Optional[PlacementGroup] = None  # Placement group for actors

    def allocate_placement_group(self, graph: DLExecutionGraph):
        """Allocate placement group for all actors.
        Each workload group will be allocated to a placement group bundle."""
        bundles: List[ResourceDesc] = []
        for group in self._config.dl_config.workload_group:
            bundle_id_start = len(bundles)
            for _ in range(group.num):
                bundles.append(group.resource)
            for workload in group.workloads:
                role = graph.roles[workload]
                for worker in role.instances:
                    worker.bundle_index = (
                        bundle_id_start + worker.rank // worker.spec.per_group
                    )
        self._pg = self._create_pg(bundles)
        assert self._pg is not None
        if not self._pg.wait(timeout_seconds=30):
            raise RuntimeError(
                f"Failed to create placement group for job {self._config.job_name}."
            )

    async def create_actors(self, graph: DLExecutionGraph):
        """Create/Get actors for all nodes in the execution graph."""
        job_info = self._config.to_job_info()
        for role in graph.roles.values():
            for worker in role.instances:
                assert (
                    self._pg is not None
                ), "Placement group must be created before creating actors."
                assert (
                    worker.bundle_index >= 0
                ), f"Worker {worker.name} bundle index must be allocated."
                spec = RayActorSpec(
                    name=worker.name,
                    resource=role.spec.instance_resource,
                    cls=role.spec.get_worker_cls(),  # type: ignore[assignment]
                    envs=worker.get_envs(),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self._pg,
                        placement_group_bundle_index=worker.bundle_index,
                        placement_group_capture_child_tasks=True,
                    ),
                    options={
                        "job_info": job_info,
                        "actor_info": worker.to_actor_info(),
                    },
                )
                self.create_actor(spec)
            if role.sub_master is not None:
                # Create sub-master node if it exists
                spec = RayActorSpec(
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
                self.create_actor(spec)
        logger.info("Finished creating nodes for the job.")

        # 2. Check actors with ping
        res: BatchInvokeResult[str] = await invoke_actors_async(
            [node.name for node in graph.vertices], "status"
        )
        logger.info(f"Actors status: {res.as_dict()}")

    def create_actor(self, node: RayActorSpec):
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
            # num_gpus=node.resource.gpu, # use bundle resource instead
            resources=node.resource.user_defined,
            runtime_env=runtime_env,
            scheduling_strategy=node.scheduling_strategy,
        ).remote(**node.options)
        logger.info(f"Created {node}")

    def _create_pg(self, bundles: List[ResourceDesc]) -> PlacementGroup:
        """Create a placement group with the given bundles."""

        accelerator = self._config.dl_config.accelerator_type

        def _to_bundle(resource: ResourceDesc) -> Dict[str, Any]:
            """Convert ResourceDesc to a bundle dict."""
            ret = {
                "CPU": resource.cpu,
                "memory": resource.memory,
            }
            if accelerator == ACCELERATOR_TYPE.GPU:
                ret["GPU"] = resource.accelerator
            elif accelerator == ACCELERATOR_TYPE.CPU:
                ret["CPU"] = max(ret["CPU"], resource.accelerator)
            return ret

        logger.info(
            f"Creating placement group for job {self._config.job_name} "
            f"with resource: {sum(bundles, ResourceDesc())} "
        )
        return placement_group(
            bundles=[_to_bundle(bundle) for bundle in bundles],
            strategy="PACK",
            name=f"dlrover_placement_group_{self._config.job_name}",
        )
