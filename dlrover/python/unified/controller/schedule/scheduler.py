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
from dlrover.python.unified.common.config import (
    ACCELERATOR_TYPE,
    JobConfig,
)
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.workload_desc import ResourceDesc
from dlrover.python.unified.util.actor_helper import wait_ready

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
        Each workload group will be allocated to a placement group bundle.
        """
        bundles: List[ResourceDesc] = []
        for group in self._config.dl_config.workload_group:
            bundle_id_start = len(bundles)
            for _ in range(group.num):
                bundles.append(group.resource)
            for workload in group.workloads:
                role = graph.roles[workload]
                for worker in role.instances:
                    worker.bundle_index = (
                        bundle_id_start + worker.rank // role.spec.per_group
                    )
        self._pg = self._create_pg(bundles)

        assert self._pg is not None
        if not self._pg.wait(timeout_seconds=30):
            raise RuntimeError(
                f"Failed to create placement group for job {self._config.job_name}."
            )

    async def create_actors(self, graph: DLExecutionGraph):
        """Create/Get actors for all actors in the execution graph."""
        job_info = self._config.to_job_info()
        global_envs = {
            DLWorkloadEnv.JOB: self._config.job_name,
            **self._config.dl_config.global_envs,
        }
        for role in graph.roles.values():
            role_envs = {
                **global_envs,
                DLWorkloadEnv.ROLE: role.name,
                **role.spec.envs,
            }
            if role.spec.rank_based_gpu_selection:
                role_envs.update(DLWorkloadEnv.RAY_NOSET_VISIBLE_DEVICES_ENVS)
            for worker in role.instances:
                assert self._pg is not None, (
                    "Placement group must be created before creating actors."
                )
                assert worker.bundle_index >= 0, (
                    f"Worker {worker.name} bundle index must be allocated."
                )
                spec = RayActorSpec(
                    name=worker.name,
                    resource=role.spec.resource,
                    cls=role.spec.get_worker_cls(),  # type: ignore[assignment]
                    envs={
                        **role_envs,
                        DLWorkloadEnv.NAME: worker.name,
                        DLWorkloadEnv.RANK: str(worker.rank),
                        DLWorkloadEnv.WORLD_SIZE: str(worker.world_size),
                        DLWorkloadEnv.LOCAL_RANK: str(worker.local_rank),
                        DLWorkloadEnv.LOCAL_WORLD_SIZE: str(
                            worker.local_world_size
                        ),
                    },
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

                # set isolation resource if required
                worker_isolation_resource = (
                    self._config.worker_isolation_schedule_resource
                )
                if worker_isolation_resource:
                    spec.resource.user_defined[worker_isolation_resource] = 1
                    logger.debug(
                        f"Setup worker actor: {worker.name} isolation "
                        f"ud resource: {worker_isolation_resource}"
                    )

                logger.info(
                    f"Creating worker actor: {worker.to_actor_info()} "
                    f"with bundle: {worker.bundle_index}"
                )
                self.create_actor(spec)
            if role.sub_master is not None:
                # Create sub-master node if it exists
                sub_master = role.sub_master
                spec = RayActorSpec(
                    name=sub_master.name,
                    # resource=role.spec.instance_resource,
                    cls=role.spec.get_master_cls(),  # type: ignore[assignment]
                    envs={
                        **role_envs,
                        DLWorkloadEnv.NAME: sub_master.name,
                    },
                    scheduling_strategy=None,  # no scheduling strategy for now
                    options={
                        "job_info": job_info,
                        "actor_info": sub_master.to_actor_info(),
                    },
                )

                # set isolation resource if required
                master_isolation_resource = (
                    self._config.master_isolation_schedule_resource
                )
                if master_isolation_resource:
                    spec.resource.user_defined[master_isolation_resource] = 1
                    logger.debug(
                        f"Setup sub-master actor: {sub_master.name} isolation "
                        f"ud resource: {master_isolation_resource}"
                    )

                logger.info(
                    f"Creating sub-master actor: {sub_master.to_actor_info()} with bundle: {sub_master.bundle_index}"
                )
                self.create_actor(spec)
        logger.info("Finished creating actors for the job.")

        # 2. Check actors with ping
        await wait_ready([node.name for node in graph.vertices])
        logger.info("All actors finished initializing.")

    def create_actor(self, actor: RayActorSpec):
        runtime_env: dict = {
            "env_vars": actor.envs,
        }
        # setup working dir
        if DLWorkloadEnv.WORKING_DIR in actor.envs:
            runtime_env["working_dir"] = actor.envs[DLWorkloadEnv.WORKING_DIR]

        if self._config.dl_config.accelerator_type == ACCELERATOR_TYPE.CPU:
            num_gpus = 0.0
        else:
            num_gpus = actor.resource.accelerator

        logger.debug(
            f"Creating actor: {actor}, "
            f"with num_gpus: {num_gpus},"
            f"with runtime env: {runtime_env}"
        )

        actor.cls.options(
            name=actor.name,
            lifetime="detached",
            max_restarts=-1,  # Allow unlimited restarts
            get_if_exists=True,
            num_cpus=actor.resource.cpu,
            memory=actor.resource.memory,
            num_gpus=num_gpus,  # use bundle resource instead
            resources=actor.resource.user_defined,
            runtime_env=runtime_env,
            scheduling_strategy=actor.scheduling_strategy,
        ).remote(**actor.options)

    def _create_pg(self, bundles: List[ResourceDesc]) -> PlacementGroup:
        """Create a placement group with the given bundles."""

        accelerator = self._config.dl_config.accelerator_type

        def _to_bundle(resource: ResourceDesc) -> Dict[str, Any]:
            """Convert ResourceDesc to a bundle dict."""
            ret = {
                "CPU": resource.cpu,
                "memory": resource.memory,
                **resource.user_defined,
            }
            if accelerator == ACCELERATOR_TYPE.GPU:
                ret["GPU"] = resource.accelerator
            elif accelerator == ACCELERATOR_TYPE.CPU:
                ret["CPU"] = max(ret["CPU"], resource.accelerator)

            # add isolation ud resource for bundles
            worker_isolation_resource = (
                self._config.worker_isolation_schedule_resource
            )
            if worker_isolation_resource:
                # a fixed value is enough
                ret[worker_isolation_resource] = 100

            # remove value=0
            return {k: v for k, v in ret.items() if v != 0}

        bundles_for_pg = [_to_bundle(bundle) for bundle in bundles]
        logger.info(
            "Creating placement group "
            f"with bundle size: {len(bundles)} "
            f"with total resource: {sum(bundles, ResourceDesc())}. \n"
            f"All bundles: {bundles_for_pg}."
        )
        return placement_group(
            bundles=bundles_for_pg,
            strategy="PACK",
            name=f"dlrover_placement_group_{self._config.job_name}",
        )
