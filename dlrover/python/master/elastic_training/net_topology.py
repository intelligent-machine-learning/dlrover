# Copyright 2024 The DLRover Authors. All rights reserved.
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
import json
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class NodeTopologyMeta(object):
    node_rank: int = 0
    process_num: int = 0
    node_ip: str = ""
    asw: str = ""
    psw: str = ""

    def __repr__(self) -> str:
        d: Dict[str, Any] = {}
        d["node_rank"] = self.node_rank
        d["process_num"] = self.process_num
        if self.node_ip:
            d["node_ip"] = self.node_ip
        if self.asw:
            d["asw"] = self.asw
        if self.psw:
            d["psw"] = self.psw
        return json.dumps(d)


class TopologyQuerier(metaclass=ABCMeta):
    @abstractmethod
    def query(self, node_ip) -> Tuple[str, str]:
        """Query the asw and psw id of a node by the IP."""
        pass


class TopologySorter(metaclass=ABCMeta):
    @abstractmethod
    def sort(
        self, nodes: Dict[int, NodeTopologyMeta]
    ) -> Dict[int, NodeTopologyMeta]:
        """Query the asw and psw id of a node by the IP."""
        pass


class DefaultTopologyQuerier(TopologyQuerier):
    def query(self, node_ip) -> Tuple[str, str]:
        return "", ""


class DpTopologySorter(TopologySorter):
    """
    The sorter places the nodes under an asw (access switch) together in
    the list of nodes. In allreduce communication, the communication packets
    between nodes with continuous ranks under an asw will not pass the psw.

    """

    def sort(
        self, nodes: Dict[int, NodeTopologyMeta]
    ) -> Dict[int, NodeTopologyMeta]:
        asw_nodes: Dict[str, List[NodeTopologyMeta]] = {}
        rank0_node = next(iter(nodes.values()))
        rank0_asw = rank0_node.asw
        for _, meta in nodes.items():
            asw_nodes.setdefault(meta.asw, [])
            asw_nodes[meta.asw].append(meta)

        sorted_nodes: Dict[int, NodeTopologyMeta] = OrderedDict()
        asw0_nodes = asw_nodes.pop(rank0_asw, [])
        for node_meta in asw0_nodes:
            sorted_nodes[node_meta.node_rank] = node_meta

        for node_metas in asw_nodes.values():
            for node_meta in node_metas:
                sorted_nodes[node_meta.node_rank] = node_meta
        return sorted_nodes
