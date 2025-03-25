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
from itertools import chain
from typing import Dict, List

from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.context import RLContext
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class RLExecutionVertex(PickleSerializable):
    def __init__(
        self,
        role: RLRoleType,
        module_name: str,
        class_name: str,
        resource: Resource,
        rank: int,
        world_size: int,
    ):
        # static info
        self.__role = role
        self.__name = role.value + "-" + str(rank)
        self.__class_obj = get_class_by_module_and_class_name(
            module_name, class_name
        )
        self.__resource = resource
        self.__rank = rank
        self.__world_size = world_size

        # runtime info
        self._create_time = 0
        self._exit_time = 0
        self._hostname = ""
        self._hostip = ""
        self._restart_count = 0

    @property
    def role(self):
        return self.__role

    @property
    def name(self):
        return self.__name

    @property
    def class_obj(self):
        return self.__class_obj

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
    def create_time(self):
        return self._create_time

    @property
    def exit_time(self):
        return self._exit_time

    @property
    def hostname(self):
        return self._hostname

    @property
    def hostip(self):
        return self._hostip

    @property
    def restart_count(self):
        return self._restart_count

    def update_runtime_info(
        self,
        create_time=None,
        exit_time=None,
        hostname=None,
        hostip=None,
        restart_count=None,
    ):
        if create_time:
            self._create_time = create_time
        if exit_time:
            self._exit_time = exit_time
        if hostname:
            self._hostname = hostname
        if hostip:
            self._hostip = hostip
        if restart_count:
            self._restart_count = restart_count


class RLExecutionEdge(PickleSerializable):
    """TODO: design and impl"""

    pass


class RLExecutionGraph(PickleSerializable):
    def __init__(self, rl_context: RLContext):
        self.__rl_context = rl_context

        self.__execution_vertices: Dict[
            RLRoleType, List[RLExecutionVertex]
        ] = {}
        self.__execution_edges: List[RLExecutionEdge] = []  # not used for now

        self._build()

    def __repr__(self):
        return (
            f"RLExecutionGraph(vertices={self.__execution_vertices}, "
            f"edges={self.__execution_edges}"
        )

    def get_rl_config(self):
        return self.__rl_context.trainer.config

    def _build(self):
        for role, desc in self.__rl_context.workloads.items():
            if desc:
                vertices = []
                for i in range(desc.instance_number):
                    vertices.append(
                        RLExecutionVertex(
                            role,
                            desc.module_name,
                            desc.class_name,
                            i,
                            desc.instance_number,
                        )
                    )
                self.__execution_vertices[role] = vertices

    @property
    def execution_vertices(self) -> Dict[RLRoleType, List[RLExecutionVertex]]:
        return self.__execution_vertices

    @property
    def execution_edges(self) -> List[RLExecutionEdge]:
        return self.__execution_edges

    def get_all_vertices(self):
        return list(chain(*self.__execution_vertices.values()))

    def get_vertices_by_role_type(
        self, role_type: RLRoleType
    ) -> List[RLExecutionVertex]:
        return self.__execution_vertices[role_type]

    def get_vertex(self, role_type: RLRoleType, rank) -> RLExecutionVertex:
        return self.__execution_vertices[role_type][rank]
