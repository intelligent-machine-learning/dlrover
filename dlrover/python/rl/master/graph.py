# Copyright 2025 The EasyDL Authors. All rights reserved.
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
import time
from itertools import chain
from typing import Dict, List, Optional

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)

from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.util.common_util import get_class_by_module_and_class_name


def get_vertex_name(
    role: RLRoleType, world_size, rank, local_world_size, local_rank
):
    return (
        role.value
        + "_"
        + str(world_size)
        + "-"
        + str(rank)
        + "_"
        + str(local_world_size)
        + "-"
        + str(local_rank)
    )


class PlacementGroupInfo(PickleSerializable):
    def __init__(self):
        self._pg_instance = None
        self._bundle_index = None

    def reset(self):
        self._pg_instance = None
        self._bundle_index = None

    @property
    def pg_instance(self):
        return self._pg_instance

    @property
    def bundle_index(self):
        return self._bundle_index

    def update_pg_instance(self, pg: PlacementGroup):
        self._pg_instance = pg

    def update_bundle_index(self, bundle_index: int):
        self._bundle_index = bundle_index


class RLExecutionVertex(PickleSerializable):
    def __init__(
        self,
        role: RLRoleType,
        module_name: str,
        class_name: str,
        resource: Resource,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
    ):
        # static info
        self.__role = role
        self.__name = get_vertex_name(
            role, world_size, rank, local_world_size, local_rank
        )
        self.__module_name = module_name
        self.__class_name = class_name
        self.__resource = resource
        self.__rank = rank
        self.__world_size = world_size
        self.__local_rank = local_rank
        self.__local_world_size = local_world_size

        # runtime info
        self._pg_info = PlacementGroupInfo()
        self._actor_handle = None
        self._create_time = 0
        self._exit_time = 0
        self._hostname = ""
        self._host_ip = ""
        self._restart_count = 0

    @property
    def role(self):
        return self.__role

    @property
    def name(self):
        return self.__name

    @property
    def module_name(self):
        return self.__module_name

    @property
    def class_name(self):
        return self.__class_name

    @property
    def resource(self):
        return self.__resource

    @property
    def rank(self):
        return self.__rank

    @property
    def world_size(self):
        return self.__world_size

    @property
    def local_rank(self):
        return self.__local_rank

    @property
    def local_world_size(self):
        return self.__local_world_size

    @property
    def pg(self):
        return self._pg_info.pg_instance

    @property
    def pg_bundle_index(self):
        return self._pg_info.bundle_index

    @property
    def actor_handle(self):
        return self._actor_handle

    @property
    def create_time(self):
        return self._create_time

    @property
    def exit_time(self):
        return self._exit_time

    @property
    def hostname(self):
        return self._hostname

    @property
    def host_ip(self):
        return self._host_ip

    @property
    def restart_count(self):
        return self._restart_count

    def get_cls(self):
        return get_class_by_module_and_class_name(
            self.module_name, self.class_name
        )

    def get_actor_id(self) -> str:
        if self._actor_handle:
            return self._actor_handle.actor_id.hex()
        return ""

    def is_pg_allocated(self) -> bool:
        return self._pg_info.pg_instance is not None

    def use_pg(self) -> bool:
        return self.is_pg_allocated() and self.pg_bundle_index is not None

    def update_pg_info(self, pg: PlacementGroup, bundle_index: int):
        self._pg_info.update_pg_instance(pg)
        self._pg_info.update_bundle_index(bundle_index)

    def update_actor_handle(self, actor_handle):
        self._actor_handle = actor_handle

    def cleanup(self):
        if self._actor_handle:
            ray.kill(self._actor_handle)

        # re-check to makesure actor is removed
        while True:
            try:
                ray.get_actor(self.name)
                time.sleep(0.1)
            except ValueError:
                break

        self.reset()

    def reset(self):
        self._pg_info.reset()
        self._actor_handle = None
        self._create_time = 0
        self._exit_time = 0
        self._hostname = ""
        self._host_ip = ""
        self._restart_count = 0

    def update_runtime_info(
        self,
        create_time=None,
        exit_time=None,
        hostname=None,
        host_ip=None,
        restart_count=None,
    ):
        if create_time:
            self._create_time = create_time
        if exit_time:
            self._exit_time = exit_time
        if hostname:
            self._hostname = hostname
        if host_ip:
            self._host_ip = host_ip
        if restart_count:
            self._restart_count = restart_count


class RLExecutionEdge(PickleSerializable):
    """TODO: design and impl"""

    pass


class PlacementGroupAllocation(PickleSerializable):
    def __init__(
        self,
        group_name,
        group_index,
        strategy,
        bundles: List[Dict[str, float]],
    ):
        self._group_name = group_name
        self._group_index = group_index
        self._strategy = strategy
        self._bundles = bundles

        # key: vertex name, value: bundle index
        self._allocation: Dict[str, int] = {}
        self._instance: Optional[PlacementGroup] = None

    @property
    def name(self):
        return self._group_name

    @property
    def pg_instance(self):
        return self._instance

    def get_bundle_resource(self) -> List[Dict[str, float]]:
        return self._bundles

    def get_bundle_index_by_vertex_name(self, vertex_name) -> int:
        if vertex_name in self._allocation:
            return self._allocation[vertex_name]
        return -1

    def get_bundle_size(self) -> int:
        return len(self._bundles)

    def is_bundle_allocated(self, bundle_index):
        return bundle_index in self._allocation.values()

    def allocate(self, vertex_name: str, bundle_index: int) -> int:
        self._allocation[vertex_name] = bundle_index
        return bundle_index

    def is_full(self):
        return len(self._allocation) >= self.get_bundle_size()

    def create_placement_group(self, timeout=10):
        if not self._instance:
            pg = placement_group(
                self.get_bundle_resource(),
                strategy=self._strategy,
                lifetime="detached",
            )
            self._instance = pg
        if self._instance:
            try:
                ray.get(self._instance.ready(), timeout=timeout)
            except GetTimeoutError as e:
                raise e

    def remove_placement_group(self):
        if self._instance:
            remove_placement_group(self._instance)


class RLExecutionGraph(PickleSerializable):
    def __init__(self, rl_context: RLContext):
        # core field
        self.__rl_context = rl_context
        self.__execution_vertices: Dict[
            RLRoleType, List[RLExecutionVertex]
        ] = {}
        self.__execution_edges: List[RLExecutionEdge] = []  # not used for now

        self.__placement_groups: Dict[str, PlacementGroupAllocation] = {}

        # mapping field(for easy using)
        self._name_vertex_mapping: Dict[str, RLExecutionVertex] = {}
        self._name_actor_mapping: Dict[str, ActorHandle] = {}

        self._build()

    def __repr__(self):
        return (
            f"RLExecutionGraph(vertices={self.__execution_vertices}, "
            f"edges={self.__execution_edges}"
        )

    @property
    def rl_context(self):
        return self.__rl_context

    @property
    def rl_config(self):
        return self.__rl_context.config

    def _build(self):
        # for role in group
        for group_desc_tuple in self.get_workload_group().groups:
            # for each group
            group_dict = group_desc_tuple[0]

            # create vertex for each group
            for role, role_group_size in group_dict.items():
                workload_desc = self.rl_context.workloads[role]
                assert workload_desc is not None

                vertices = []
                for i in range(workload_desc.instance_number):
                    vertex = RLExecutionVertex(
                        role,
                        workload_desc.module_name,
                        workload_desc.class_name,
                        workload_desc.instance_resource,
                        rank=i,
                        world_size=workload_desc.instance_number,
                        local_rank=i % role_group_size,
                        local_world_size=role_group_size,
                    )

                    vertices.append(vertex)
                    self._name_vertex_mapping[vertex.name] = vertex

                self.__execution_vertices[role] = vertices

    @property
    def execution_vertices(self) -> Dict[RLRoleType, List[RLExecutionVertex]]:
        return self.__execution_vertices

    @property
    def execution_edges(self) -> List[RLExecutionEdge]:
        return self.__execution_edges

    @property
    def name_vertex_mapping(self) -> Dict[str, RLExecutionVertex]:
        return self._name_vertex_mapping

    @property
    def name_actor_mapping(self) -> Dict[str, ActorHandle]:
        return self._name_actor_mapping

    def get_all_vertices(self) -> List[RLExecutionVertex]:
        return list(chain(*self.__execution_vertices.values()))

    def get_vertices_by_role_type(
        self, role_type: RLRoleType
    ) -> List[RLExecutionVertex]:
        if role_type in self.execution_vertices:
            return self.execution_vertices[role_type]
        return []

    def get_vertex(self, role_type: RLRoleType, rank) -> RLExecutionVertex:
        return self.__execution_vertices[role_type][rank]

    def get_all_actor_handles(self) -> List[ActorHandle]:
        return [vertex.actor_handle for vertex in self.get_all_vertices()]

    def get_actor_handles(self) -> Dict[RLRoleType, List[ActorHandle]]:
        return {
            role: [vertex.actor_handle for vertex in vertices]
            for role, vertices in self.__execution_vertices.items()
        }

    def get_actor_cls(self) -> Dict[RLRoleType, type]:
        return {
            role: vertices[0].get_cls()
            for role, vertices in self.__execution_vertices.items()
        }

    def get_trainer_cls(self):
        return get_class_by_module_and_class_name(
            self.rl_context.trainer.module_name,
            self.rl_context.trainer.class_name,
        )

    def get_workload_group(self):
        return self.rl_context.workload_group

    def get_unit_resource_by_role(
        self, role: RLRoleType
    ) -> Optional[Resource]:
        vertices_by_role = self.get_vertices_by_role_type(role)
        if vertices_by_role:
            return vertices_by_role[0].resource
        return None

    def get_workloads_size_by_role(self, role: RLRoleType) -> int:
        return len(self.get_vertices_by_role_type(role))

    def update_actor_handle_for_vertex(
        self, actor_handle: ActorHandle, vertex: RLExecutionVertex
    ):
        vertex.update_actor_handle(actor_handle)
        self._name_actor_mapping[vertex.name] = actor_handle

    def add_placement_group(self, pg_name, pg):
        self.__placement_groups[pg_name] = pg

    def get_placement_group(self, name=None):
        if name:
            return self.__placement_groups[name]
        return self.__placement_groups

    def create_placement_group(self):
        for pg_allocation in self.__placement_groups.values():
            pg_allocation.create_placement_group()

    def remove_placement_group(self):
        for pg_allocation in self.__placement_groups.values():
            pg_allocation.remove_placement_group()

    def cleanup_placement_group_allocation(self):
        self.__placement_groups = {}

    def get_bundle_topology(self):
        """Return topology in format: [${vertex_num}]"""
        topology = []
        for group_desc_tuple in self.get_workload_group().groups:
            group_size = sum(group_desc_tuple[0].values())
            for _ in range(group_desc_tuple[1]):
                topology.append(group_size)

        return topology
