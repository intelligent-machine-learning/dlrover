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
from typing import Dict, Optional, Union

from dlrover.python.common.constants import NodeType, PreCheckStatus
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisActionQueue,
)

_dlrover_context = Context.singleton_instance()


class JobContext(Singleton):
    """
    JobContext includes critical states of the training job that
    will be shared across multiple components.
    """

    def __init__(self):
        self._action_queue = DiagnosisActionQueue()
        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._total_worker_num = 0
        self._failed_nodes: Dict[int, int] = {}
        self._pre_check_status: str = PreCheckStatus.CHECKING
        self._locker = threading.Lock()

    def enqueue_actions(self, actions):
        for action in actions:
            self.enqueue_action(action)

    def enqueue_action(self, action):
        if not action or action.action_type == DiagnosisActionType.NONE:
            return
        self._action_queue.add_action(action)

    def next_action(
        self,
        instance=DiagnosisConstant.MASTER_INSTANCE,
    ):
        return self._action_queue.next_action(instance=instance)

    def get_mutable_ps_nodes(self):
        return self.get_mutable_job_nodes(NodeType.PS)

    def get_mutable_worker_nodes(self):
        return self.get_mutable_job_nodes(NodeType.WORKER)

    def get_mutable_job_nodes(self, node_type) -> Dict[int, Node]:
        with self._locker:
            if node_type in self._job_nodes:
                return self._job_nodes[node_type]
            return {}

    def job_nodes(self) -> Dict[str, Dict[int, Node]]:
        """Get global job nodes dict

        Returns:
            return global _job_nodes dict

        The caller should use self._locker to synchronize
        """
        return self._job_nodes

    def dup_job_nodes(self) -> Dict[str, Dict[int, Node]]:
        """Get global job nodes dict

        Returns:
            return global _job_nodes dict
        """
        with self._locker:
            return copy.deepcopy(self._job_nodes)

    def job_nodes_by_type(self, node_type: str) -> Dict[int, Node]:
        """Get nodes list by type

        Args:
            node_type: node type

        Returns:
            node list of the node_type in global _job_nodes dict

        The caller should use self._locker to synchronize
        """
        node_type = self._preprocess(node_type)
        if node_type not in self._job_nodes:
            return {}
        return self._job_nodes[node_type]

    def dup_job_nodes_by_type(self, node_type: str) -> Dict[int, Node]:
        """Get nodes list by type

        Args:
            node_type: node type

        Returns:
            node list of the node_type in global _job_nodes dict
        """
        with self._locker:
            node_type = self._preprocess(node_type)
            if node_type not in self._job_nodes:
                return {}
            return copy.deepcopy(self._job_nodes[node_type])

    def job_node(self, node_type: str, node_id: int) -> Optional[Node]:
        """Get node by type and id

        Args:
            node_type: node type
            node_id: node id

        Returns:
            Node or None if node does not exist

        The caller should use self._locker to synchronize
        """
        node_type = self._preprocess(node_type)
        if (
            node_type not in self._job_nodes
            or node_id not in self._job_nodes[node_type]
        ):
            return None
        return self._job_nodes[node_type][node_id]

    def dup_job_node(self, node_type: str, node_id: int) -> Optional[Node]:
        """Get deepcopy of node by type and id

        Args:
            node_type: node type
            node_id: node id

        Returns:
            Node or None if node does not exist
        """
        with self._locker:
            node_type = self._preprocess(node_type)
            if (
                node_type not in self._job_nodes
                or node_id not in self._job_nodes[node_type]
            ):
                return None
            return copy.deepcopy(self._job_nodes[node_type][node_id])

    def _preprocess(self, node_type: str) -> str:
        if node_type == NodeType.CHIEF and node_type not in self._job_nodes:
            return NodeType.MASTER
        return node_type

    def update_job_nodes_by_type(self, node_type, job_nodes: Dict[int, Node]):
        with self._locker:
            if self._job_nodes is None:
                self._job_nodes = {}
            if node_type not in self._job_nodes:
                self._job_nodes[node_type] = {}
            self._job_nodes[node_type] = copy.deepcopy(job_nodes)

    def update_job_nodes(self, job_nodes: Dict[str, Dict[int, Node]]):
        with self._locker:
            self._job_nodes = copy.deepcopy(job_nodes)

    def update_job_node(self, node: Node):
        with self._locker:
            if self._job_nodes is None:
                self._job_nodes = {}
            if node.type not in self._job_nodes:
                self._job_nodes[node.type] = {}
            self._job_nodes[node.type][node.id] = copy.deepcopy(node)

    def clear_job_nodes(self):
        with self._locker:
            self._job_nodes = {}

    def report_failed_node(self, node_id: Union[int, str] = None):
        if node_id is None:
            return

        node_id = int(node_id)
        with self._locker:
            if node_id not in self._failed_nodes:
                self._failed_nodes[node_id] = int(time.time())

    def get_failed_node_cnt(self):
        return len(self._failed_nodes)

    def set_pre_check_status(self, status: str):
        self._pre_check_status = status

    def get_pre_check_status(self):
        if _dlrover_context.pre_check_enabled:
            return self._pre_check_status
        return PreCheckStatus.DISABLED

    def update_total_worker_num(self, worker_num: int):
        self._total_worker_num = worker_num

    def get_total_worker_num(self):
        return self._total_worker_num


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context
