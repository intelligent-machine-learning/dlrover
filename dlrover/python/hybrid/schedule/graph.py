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
from dataclasses import dataclass
from typing import Dict, List, Optional

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dlrover.python.common.resource import Resource
from dlrover.python.hybrid.config import DLConfig, WorkloadDesc
from dlrover.python.hybrid.schedule.scheduler import Node
from dlrover.python.hybrid.worker.worker import Worker
from dlrover.python.unified.common.constant import DLWorkloadEnv


@dataclass
class PlacementGroupSpec:
    name: str
    strategy: PlacementGroupSchedulingStrategy
    bundles: List[Resource]


class DLExecutionGraph:
    """The Logical computational graph for distributed deep learning."""

    @dataclass
    class Vertex:
        """
        Vertex expression for computational graph.

        role: Role of the vertex.
        module_name: Module name of the vertex's class.
        class_name: Class name of the vertex's class.
        resource: Resource the vertex required.
        """

        role: str
        spec: WorkloadDesc

        world_size: int
        rank: int
        local_world_size: int
        local_rank: int

        placement_group: Optional[PlacementGroupSpec] = None
        bundle_index: int = -1

        @property
        def name(self):
            return f"{self.role}_{self.world_size}-{self.rank}_{self.local_world_size}-{self.local_rank}"

        def get_envs(self) -> Dict[str, str]:
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
            if not set(DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS.items()).issubset(
                set(envs.items())
            ):
                # this env is used for disable 'ray set visible device' so we can
                # specify device by local_rank on ray(otherwise ray will assign a
                # specified device)
                envs.update(DLWorkloadEnv.RAY_NOSET_VISIBLE_DEVICES_ENVS)
            else:
                # remove 'false' value setting for using 'ray set visible device'
                for key in DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS:
                    envs.pop(key, None)

            return envs

    @dataclass
    class Edge:
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

    def __init__(self, vertices: List[Vertex], edges: List[Edge]):
        self.vertices = vertices
        self.edges = edges

    @classmethod
    def create(cls, dl_config: DLConfig) -> "DLExecutionGraph":
        vertices = []
        edges = []
        for name, workload in dl_config.workloads.items():
            for i in range(workload.instance_number):
                vertex = cls.Vertex(
                    role=name,
                    spec=workload,
                    world_size=workload.instance_number,
                    rank=i,
                    local_world_size=workload.per_node,
                    local_rank=i % workload.per_node,
                )
                vertices.append(vertex)
        return cls(vertices, edges)

    def prepare_nodes(self) -> List[Node]:
        nodes = []
        for vertex in self.vertices:
            node = Node(
                name=vertex.name,
                resource=vertex.spec.instance_resource,
                envs=vertex.get_envs(),
                cls=Worker,  # type: ignore[assignment]
                kind="worker",  # all nodes are worker
                scheduling_strategy=None,  # no scheduling strategy for now
                options={},  # options is used for actor's kwargs
            )
            nodes.append(node)
        return nodes
