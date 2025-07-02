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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel

from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.workload_base import ActorInfo
from dlrover.python.unified.common.workload_config import ResourceDesc
from dlrover.python.unified.contoller.config import DLConfig, WorkloadDesc


@dataclass
class PlacementGroupSpec:
    name: str
    strategy: str  # VALID_PLACEMENT_GROUP_STRATEGIES
    bundles: List[ResourceDesc]


class DLExecutionVertex(ABC, BaseModel):
    """
    Vertex expression for computational graph.

    role: Role of the vertex.
    module_name: Module name of the vertex's class.
    class_name: Class name of the vertex's class.
    resource: Resource the vertex required.
    """

    role: str
    spec: WorkloadDesc

    placement_group: Optional[PlacementGroupSpec] = None
    bundle_index: int = -1

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the vertex.

        Returns:
            The name of the vertex.
        """

    @abstractmethod
    def get_envs(self) -> Dict[str, str]:
        """
        Get environment variables for the vertex.

        Returns:
            A dictionary of environment variables.
        """

    @abstractmethod
    def to_actor_info(self) -> "ActorInfo":
        """Convert to NodeInfo. Exposed to workers and sub-masters."""


class DLExecutionWorkerVertex(DLExecutionVertex):
    world_size: int
    rank: int
    local_world_size: int
    local_rank: int

    placement_group: Optional[PlacementGroupSpec] = None
    bundle_index: int = -1

    @property
    def name(self):
        return (
            f"{self.role}_{self.world_size}-{self.rank}"
            f"_{self.local_world_size}-{self.local_rank}"
        )

    def get_envs(self) -> Dict[str, str]:
        # TODO Do we need pass Env for workers, or set in workers themselves?
        envs = {
            # DLWorkloadEnv.JOB: _job_ctx.job_config.job_name,
            DLWorkloadEnv.NAME: self.name,
            DLWorkloadEnv.ROLE: self.role,
            DLWorkloadEnv.RANK: str(self.rank),
            DLWorkloadEnv.WORLD_SIZE: str(self.world_size),
            DLWorkloadEnv.LOCAL_RANK: str(self.local_rank),
            DLWorkloadEnv.LOCAL_WORLD_SIZE: str(self.local_world_size),
        }
        # setup global env
        # envs.update(self.graph.dl_context.env)

        # setup role env
        envs.update(self.spec.instance_env)

        # # setup device collocation env
        # if self.get_core_resource_num() < 1:
        #     group_env_value = ""
        #     for group_tuple in self.graph.dl_context.workload_group.groups:
        #         group_roles = list(group_tuple[0].keys())
        #         if self.role in group_roles:
        #             group_env_value = ",".join(r for r in group_roles)
        #             break
        #     envs[DLWorkloadEnv.DEVICE_COLLOCATION_GROUP] = group_env_value

        # setup ray cuda visible env
        if not set(
            DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS.items()
        ).issubset(set(envs.items())):
            # this env is used for disable 'ray set visible device' so we can
            # specify device by local_rank on ray(otherwise ray will assign a
            # specified device)
            envs.update(DLWorkloadEnv.RAY_NOSET_VISIBLE_DEVICES_ENVS)
        else:
            # remove 'false' value setting for using 'ray set visible device'
            for key in DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS:
                envs.pop(key, None)

        return envs

    def to_actor_info(self) -> "ActorInfo":
        return ActorInfo(
            name=self.name,
            role=self.role,
            spec=self.spec,
            rank=self.rank,
            local_rank=self.local_rank,
        )


class DLExecutionMasterVertex(DLExecutionVertex):
    """
    Master vertex in the computational graph.

    role: Role of the master vertex.
    spec: Workload specification for the master vertex.
    """

    @property
    def name(self):
        return f"{self.role}-master"

    def get_envs(self) -> Dict[str, str]:
        envs = {
            DLWorkloadEnv.NAME: self.name,
            DLWorkloadEnv.ROLE: self.role,
        }
        # setup global env
        # envs.update(self.graph.dl_context.env)

        # setup role env
        envs.update(self.spec.instance_env)
        return envs

    def to_actor_info(self) -> "ActorInfo":
        return ActorInfo(
            name=self.name,
            role=self.role,
            spec=self.spec,
        )


@dataclass
class DLExecutionEdge:
    """
    Edge expression for computational graph.

    index: The id of the edge in order.
    from_role: Role of the caller.
    to_role: Role of the callee.
    invocation_name: Remote function name of the invocation.
    async_group: Edges defined within the group can be invoked asynchronously
        at the same time.
    """

    from_role: str
    to_role: str
    invocation_name: str
    async_group: Optional[str] = None


@dataclass
class DLWorkloadRole:
    name: str
    spec: WorkloadDesc
    instance_number: int

    def __post_init__(self):
        self.instances = [
            DLExecutionWorkerVertex(
                role=self.name,
                spec=self.spec,
                world_size=self.instance_number,
                rank=i,
                local_world_size=self.spec.per_node,
                local_rank=i % self.spec.per_node,
            )
            for i in range(self.instance_number)
        ]
        self.sub_master: Optional[DLExecutionMasterVertex] = None
        if self.spec.get_master_cls() is not None:
            self.sub_master = DLExecutionMasterVertex(
                role=self.name,
                spec=self.spec,
            )


class DLExecutionGraph:
    """The computational graph for distributed deep learning."""

    def __init__(
        self, roles: Dict[str, DLWorkloadRole], edges: List[DLExecutionEdge]
    ):
        self.roles = roles
        self.edges = edges

        # note: vertices includes both worker and sub-master vertices
        self.vertices: List[DLExecutionVertex] = []
        self.by_name: Dict[str, DLExecutionVertex] = {}

        self.build_cache()

    def build_cache(self):
        """Build cache for quick access."""
        self.vertices = [
            vertex for role in self.roles.values() for vertex in role.instances
        ]
        self.vertices.extend(
            role.sub_master for role in self.roles.values() if role.sub_master
        )
        self.by_name = {vertex.name: vertex for vertex in self.vertices}

    @classmethod
    def create(cls, dl_config: DLConfig) -> "DLExecutionGraph":
        roles = {
            name: DLWorkloadRole(
                name=name,
                spec=workload,
                instance_number=workload.instance_number,
            )
            for name, workload in dl_config.workloads.items()
        }
        return cls(roles, edges=[])
