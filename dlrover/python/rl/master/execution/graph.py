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

from rl.common.context import RLContext

from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class RLExecutionVertex(PickleSerializable):
    def __init__(
        self, role: RLRoleType, module_name, class_name, rank, world_size
    ):
        self.__role = role
        self.__class_obj = get_class_by_module_and_class_name(
            module_name, class_name
        )
        self.__rank = rank
        self.__world_size = world_size

    @property
    def role(self):
        return self.__role

    @property
    def class_obj(self):
        return self.__class_obj

    @property
    def rank(self):
        return self.__rank

    @property
    def world_size(self):
        return self.__world_size


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
