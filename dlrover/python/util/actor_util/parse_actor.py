# Copyright 2023 The DLRover Authors. All rights reserved.
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

from dlrover.python.common.constants import NodeType


def parse_type(name) -> str:
    name = name.lower()
    node_type: str = ""
    if NodeType.PS in name:
        node_type = NodeType.PS
    elif NodeType.EVALUATOR in name:
        node_type = NodeType.EVALUATOR
    elif NodeType.WORKER in name:
        node_type = NodeType.WORKER
    return node_type


def parse_index(name) -> int:
    """
    PsActor_1 split("_")[-1]
    TFSinkFunction-4|20 split("|").split("-")[-1]
    """
    node_type = parse_type(name)
    node_index: int = 0
    if node_type == NodeType.PS:
        node_index = int(name.split("_")[-1])
    elif node_type == NodeType.EVALUATOR:
        node_index = 1
    elif node_type == NodeType.WORKER:
        node_index = int(name.split("|")[0].split("-")[-1])
    return node_index


def parse_type_id_from_actor_name(name):
    node_type = parse_type(name)
    node_index = parse_index(name)
    return node_type, node_index
