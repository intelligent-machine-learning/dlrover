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

import copy
import threading
import time
from typing import Dict, Optional

from dlrover.python.common.constants import NodeType
from dlrover.python.common.node import Node
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisActionQueue
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisActionConstants,
)
from datetime import datetime


class JobContext(Singleton):
    """
    JobContext includes critical states of the training job that
    will be shared across multiple components.
    """

    def __init__(self):
        self._action_queue = DiagnosisActionQueue()
        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._ps_nodes: Dict[int, Node] = {}
        self._workers: Dict[int, Node] = {}
        self._locker = threading.Lock()

    def enqueue_actions(self, actions):
        for action in actions:
            self._action_queue.add_action(action)

    def next_actions(
            self, instance=DiagnosisConstant.LOCAL_INSTANCE, action_type=DiagnosisActionConstants.ACTION_TYPE_ANY
    ):
        return self._action_queue.next_actions(instance=instance, action_type=action_type)

    def _update_job_nodes(self, job_nodes: Dict[str, Dict[int, Node]]):
        with self._locker:
            self._job_nodes = copy.deepcopy(job_nodes)
            if NodeType.PS in self._job_nodes:
                self._ps_nodes = copy.deepcopy(self._job_nodes[NodeType.PS])
            else:
                self._ps_nodes = {}

            if NodeType.WORKER in self._job_nodes:
                self._workers = copy.deepcopy(self._job_nodes[NodeType.WORKER])
            else:
                self._workers = {}

    @property
    def ps_nodes(self) -> Dict[int, Node]:
        with self._locker:
            return self._ps_nodes

    @property
    def workers(self) -> Dict[int, Node]:
        with self._locker:
            return self._workers

    def job_nodes(self) -> Dict[str, Dict[int, Node]]:
        """
        return a copy of job nodes
        """
        with self._locker:
            return copy.deepcopy(self._job_nodes)

    def job_node(self, node_type: str, node_id: int) -> Optional[Node]:
        with self._locker:
            node_type = self._preprocess(node_type)
            if (
                node_type not in self._job_nodes
                or node_id not in self._job_nodes[node_type]
            ):
                return None
            return copy.deepcopy(self._job_nodes[node_type][node_id])

    def job_nodes_by_type(self, node_type: str) -> Dict[int, Node]:
        with self._locker:
            node_type = self._preprocess(node_type)
            if node_type not in self._job_nodes:
                return {}
            return copy.deepcopy(self._job_nodes[node_type])

    def _preprocess(self, node_type: str) -> str:
        if node_type == NodeType.CHIEF and node_type not in self._job_nodes:
            return NodeType.MASTER
        return node_type

    def _update_job_node(self, node: Node):
        with self._locker:
            if self._job_nodes is None:
                self._job_nodes = {}
            if node.type not in self._job_nodes:
                self._job_nodes[node.type] = {}

            self._job_nodes[node.type][node.id] = copy.deepcopy(node)

            if node.type == NodeType.PS:
                if node.id not in self._ps_nodes:
                    self._ps_nodes[node.id] = copy.deepcopy(node)
                else:
                    self._ps_nodes[node.id].update_from_node(node)

            if node.type == NodeType.WORKER:
                if node.id not in self._workers:
                    self._workers[node.id] = copy.deepcopy(node)
                else:
                    self._workers[node.id].update_from_node(node)

    def _clear_nodes(self):
        with self._locker:
            self._job_nodes = {}
            self._ps_nodes = {}
            self._workers = {}


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context


def update_job_nodes(job_nodes: Dict[str, Dict[int, Node]]):
    job_context = JobContext.singleton_instance()
    job_context._update_job_nodes(copy.deepcopy(job_nodes))


def update_job_node(node: Node):
    if node is None:
        return
    job_context = JobContext.singleton_instance()
    job_context._update_job_node(node)


def clear_job_nodes():
    job_context = JobContext.singleton_instance()
    job_context._clear_nodes()
