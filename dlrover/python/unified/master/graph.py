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
import time
from itertools import chain
from typing import Dict, List, Optional, Tuple, Type, Union

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
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.util.common_util import get_class_by_module_and_class_name


def get_vertex_name(role: str, world_size, rank, local_world_size, local_rank):
    return (
        role
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


class FunctionInfo(object):
    """
    To express each function.
    """

    def __init__(self, name, input_func, inputs=None, outputs=None):
        # unique id
        self._name: str = name

        # input and outputs(by 'DataTransfer')
        # function impl to get inputs from 'DataTransfer'
        self._input_func = input_func
        self._inputs: List[Tuple[str, Type]] = inputs
        self._outputs: List[Tuple[str, Type]] = outputs

    @property
    def name(self) -> str:
        return self._name


class VertexInvocationMeta(object):
    """
    To express the function hold by the vertex.
    """

    def __init__(self, funcs):
        self._funcs: Dict[str, FunctionInfo] = funcs

    def get_func(self, name):
        return self._funcs[name]


class DLExecutionVertex(PickleSerializable):
    """
    Vertex expression for computational graph.

    role: Role of the vertex.
    module_name: Module name of the vertex's class.
    class_name: Class name of the vertex's class.
    resource: Resource the vertex required.
    """

    def __init__(
        self,
        role: str,
        module_name: str,
        class_name: str,
        resource: Resource,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        sub_stage: int = 0,
        sub_stage_index: int = 0,
        invocation_meta: VertexInvocationMeta = None,
        **kwargs,
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
        self.__sub_stage = sub_stage
        self.__sub_stage_index = sub_stage_index
        self.__invocation_meta = invocation_meta
        self.__kwargs = kwargs

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
    def sub_stage(self):
        return self.__sub_stage

    @property
    def sub_stage_index(self):
        return self.__sub_stage_index

    @property
    def invocation_meta(self):
        return self.__invocation_meta

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

    def get_extra_args(self, key, default_value=None):
        if key in self.__kwargs:
            return self.__kwargs[key]
        return default_value

    def get_cls(self):
        return get_class_by_module_and_class_name(
            self.module_name, self.class_name
        )

    def get_actor_id(self) -> str:
        if self._actor_handle:
            return self._actor_handle.actor_id.hex()
        return ""

    def get_core_resource_num(self):
        if self.resource.gpu > 0:
            return self.resource.gpu

        return self.resource.cpu

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


class DLExecutionEdge(PickleSerializable):
    """
    Edge expression for computational graph.

    index: The id of the edge in order.
    from_role: Role of the caller.
    to_role: Role of the callee.
    invocation_name: Remote function name of the invocation.
    async_group: Edges defined within the group can be invoked asynchronously
        at the same time.
    """

    def __init__(
        self, index, from_role, to_role, invocation_name, async_group=None
    ):
        self._index = index
        self._from_role: Union[None, str] = from_role
        self._to_role: str = to_role
        self._invocation_name: str = invocation_name
        self._async_group = async_group

    @property
    def index(self):
        return self._index

    @property
    def from_role(self):
        return self._from_role

    @property
    def to_role(self):
        return self._to_role

    @property
    def invocation_name(self):
        return self._invocation_name

    @property
    def async_group(self):
        return self._async_group


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

    def create_placement_group(self, timeout=30):
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


class DLExecutionGraph(PickleSerializable):
    def __init__(self, dl_context: DLContext):
        # core field
        self.__dl_context = dl_context
        self.__execution_vertices: Dict[str, List[DLExecutionVertex]] = {}
        # TODO: implement edge building for a fully computational graph
        # format: [[edge0], [edge1, edge2], [edge3]]
        self.__execution_edges: List[List[DLExecutionEdge]] = []

        self.__placement_groups: Dict[str, PlacementGroupAllocation] = {}

        # mapping field(for easy using)
        self._name_vertex_mapping: Dict[str, DLExecutionVertex] = {}
        self._name_actor_mapping: Dict[str, ActorHandle] = {}

        self._build()

    def __repr__(self):
        return (
            f"RLExecutionGraph(vertices={self.__execution_vertices}, "
            f"edges={self.__execution_edges}"
        )

    @property
    def dl_context(self):
        return self.__dl_context

    @property
    def dl_config(self):
        return self.__dl_context.config

    def _build(self):
        # for role in group
        for group_desc_tuple in self.get_workload_group().groups:
            # for each group
            group_dict = group_desc_tuple[0]

            # create vertex for each group
            for role, role_group_size in group_dict.items():
                workload_desc = self.dl_context.workloads[role]
                assert workload_desc is not None

                vertices = []
                for i in range(workload_desc.instance_number):
                    vertex = DLExecutionVertex(
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
    def execution_vertices(self) -> Dict[str, List[DLExecutionVertex]]:
        return self.__execution_vertices

    @property
    def execution_edges(self) -> List[List[DLExecutionEdge]]:
        return self.__execution_edges

    @property
    def name_vertex_mapping(self) -> Dict[str, DLExecutionVertex]:
        return self._name_vertex_mapping

    @property
    def name_actor_mapping(self) -> Dict[str, ActorHandle]:
        return self._name_actor_mapping

    def get_all_vertices(self) -> List[DLExecutionVertex]:
        return list(chain(*self.__execution_vertices.values()))

    def get_vertices_by_role_type(
        self, role_type: str
    ) -> List[DLExecutionVertex]:
        if role_type in self.execution_vertices:
            return self.execution_vertices[role_type]
        return []

    def get_vertex(self, role_type: str, rank) -> DLExecutionVertex:
        return self.__execution_vertices[role_type][rank]

    def get_all_actor_handles(self) -> List[ActorHandle]:
        return [vertex.actor_handle for vertex in self.get_all_vertices()]

    def get_actor_handles(self) -> Dict[str, List[ActorHandle]]:
        return {
            role: [vertex.actor_handle for vertex in vertices]
            for role, vertices in self.__execution_vertices.items()
        }

    def get_actor_metas(self) -> Dict[str, Tuple[type, float]]:
        """
        meta: class, key_resource
        """
        return {
            role: (vertices[0].get_cls(), vertices[0].get_core_resource_num())
            for role, vertices in self.__execution_vertices.items()
        }

    def get_trainer_cls(self):
        return get_class_by_module_and_class_name(
            self.dl_context.trainer.module_name,
            self.dl_context.trainer.class_name,
        )

    def get_workload_group(self):
        return self.dl_context.workload_group

    def get_unit_resource_by_role(self, role: str) -> Optional[Resource]:
        vertices_by_role = self.get_vertices_by_role_type(role)
        if vertices_by_role:
            return vertices_by_role[0].resource
        return None

    def get_workloads_size_by_role(self, role: str) -> int:
        return len(self.get_vertices_by_role_type(role))

    def update_actor_handle_for_vertex(
        self, actor_handle: ActorHandle, vertex: DLExecutionVertex
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
