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

from dlrover.python.unified.common.actor_base import ActorInfo, NodeInfo
from dlrover.python.unified.common.config import DLConfig, WorkloadDesc


class DLExecutionVertex(ABC, BaseModel):
    """
    Vertex expression for computational graph.

    role: Role of the vertex.
    module_name: Module name of the vertex's class.
    class_name: Class name of the vertex's class.
    resource: Resource the vertex required.
    """

    role: "DLWorkloadRole"

    bundle_index: int = -1
    total_failure_count: int = 0
    per_node_failure_count: int = 0
    restart_count: int = 0
    restarting: bool = False
    node_info: Optional[NodeInfo] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the vertex.

        Returns:
            The name of the vertex.
        """

    @property
    def spec(self) -> WorkloadDesc:
        """Get the workload specification of the vertex."""
        return self.role.spec

    @property
    def node(self) -> Optional[NodeInfo]:
        """Get the actor's node info."""
        return self.node_info

    @abstractmethod
    def to_actor_info(self) -> "ActorInfo":
        """Convert to NodeInfo. Exposed to workers and sub-masters."""

    def set_node(self, node_info: NodeInfo):
        self.node_info = node_info

    def inc_failure_count(self):
        self.total_failure_count += 1
        self.per_node_failure_count += 1

    def reset_per_node_failure_count(self):
        self.per_node_failure_count = 0

    def inc_restart_count(self):
        self.restart_count += 1

    def reset_restart_count(self):
        self.restart_count = 0

    def set_restarting(self):
        self.restarting = True

    def set_running(self):
        self.restarting = False


class DLExecutionWorkerVertex(DLExecutionVertex):
    """Worker vertex in the computational graph."""

    node_rank: int
    world_size: int
    rank: int
    local_world_size: int
    local_rank: int

    bundle_index: int = -1

    @property
    def name(self):
        return (
            f"{self.role.name}_{self.world_size}-{self.rank}"
            f"_{self.local_world_size}-{self.local_rank}"
        )

    def to_actor_info(self) -> "ActorInfo":
        return ActorInfo(
            name=self.name,
            role=self.role.name,
            spec=self.role.spec,
            sub_master=(
                self.role.sub_master.name if self.role.sub_master else None
            ),
            node_rank=self.node_rank,
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
        return f"{self.role.name}-master"

    def to_actor_info(self) -> "ActorInfo":
        return ActorInfo(
            name=self.name,
            role=self.role.name,
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
                role=self,
                node_rank=i // self.spec.per_group,
                world_size=self.instance_number,
                rank=i,
                local_world_size=self.spec.per_group,
                local_rank=i % self.spec.per_group,
            )
            for i in range(self.instance_number)
        ]
        self.sub_master: Optional[DLExecutionMasterVertex] = None
        if len(self.instances) > 0 and self.spec.get_master_cls() is not None:
            self.sub_master = DLExecutionMasterVertex(
                role=self,
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
                instance_number=workload.total,
            )
            for name, workload in dl_config.workloads.items()
        }
        return cls(roles, edges=[])

    def get_all_actors_by_specified_node_actors(
        self, actors: List[DLExecutionVertex]
    ):
        return self.get_all_actors_by_specified_nodes(
            set(
                [
                    actor.node_info.id
                    for actor in actors
                    if actor.node_info is not None
                ]
            )
        )

    def get_all_actors_by_specified_nodes(self, nodes: set[str]):
        return [
            actor
            for actor in self.vertices
            if actor.node_info is not None and actor.node_info.id in nodes
        ]

    def update_actor_failure(self, name):
        self.by_name[name].inc_failure_count()
