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

from dlrover.python.common.constants import (
    JobStage,
    NodeStatus,
    NodeType,
    PreCheckStatus,
)
from dlrover.python.common.global_context import (
    Context,
    DefaultValues,
)
from dlrover.python.common.log import default_logger as logger
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
        self._exit_code = 0
        self._exit_reason = None

        self._action_queue = DiagnosisActionQueue()
        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._job_restart_count = 0
        self._total_worker_num = 0
        self._failed_nodes: Dict[int, int] = {}

        self._pre_check_status: str = PreCheckStatus.CHECKING

        self._locker = threading.Lock()
        self._job_stage: str = JobStage.JOB_INIT
        self._job_pre_status: str = JobStage.JOB_INIT

        # job node groups are different groups of WORKER node
        self._group_locker = threading.Lock()
        # _job_node_groups is a dict with group id as key
        # the value is a dict with rank_index as key
        self._job_node_groups: Dict[int, Dict[int, Node]] = {}
        self._max_group_idx = DefaultValues.FIRST_GROUP_IDX

    def get_exit_code(self):
        return self._exit_code

    def get_exit_reason(self):
        return self._exit_reason

    def set_exit_code(self, exit_code):
        self._exit_code = exit_code

    def set_exit_reason(self, exit_reason):
        self._exit_reason = exit_reason

    def next_group_idx(self):
        with self._locker:
            self._max_group_idx += 1
            return self._max_group_idx

    def get_job_stage(self):
        with self._locker:
            return self._job_stage

    def update_job_stage(self, stage):
        with self._locker:
            self._job_stage = stage

    def enqueue_actions(self, actions):
        for action in actions:
            self.enqueue_diagnosis_action(action)

    def enqueue_diagnosis_action(self, action):
        if not action or action.action_type == DiagnosisActionType.NONE:
            return
        self._action_queue.add_action(action)

    def next_action(
        self,
        instance=DiagnosisConstant.MASTER_INSTANCE,
    ):
        return self._action_queue.next_action(instance=instance)

    def clear_actions(self):
        self._action_queue.clear()

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

    def job_node_by_rank(self, node_type: str, rank: int) -> Optional[Node]:
        """Get node by type and rank

        Args:
            node_type: node type
            rank: rank index

        Returns:
            Node or None if node does not exist

        The caller should use self._locker to synchronize
        """
        node_type = self._preprocess(node_type)
        if node_type not in self._job_nodes:
            return None

        for node in self._job_nodes[node_type].values():
            if node.rank_index == rank and node.status == NodeStatus.RUNNING:
                return node

        return None

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

    def job_node_groups(self) -> Dict[int, Dict[int, Node]]:
        with self._group_locker:
            return self._job_node_groups

    def job_node_groups_keys(self):
        with self._group_locker:
            return self._job_node_groups.keys()

    def job_node_group(self, node_group: int) -> Dict[int, Node]:
        with self._group_locker:
            if node_group not in self._job_node_groups:
                return {}
            return self._job_node_groups[node_group]

    def job_group_node_by_rank(
        self, node_group: int, node_rank: int
    ) -> Optional[Node]:
        with self._group_locker:
            if (
                node_group not in self._job_node_groups
                or node_rank not in self._job_node_groups[node_group]
            ):
                return None
            return self._job_node_groups[node_group][node_rank]

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

    def update_job_node_by_group(self, node: Node):
        with self._group_locker:
            if node.group not in self._job_node_groups:
                logger.info(
                    f"New node group {node.group} with size {node.group_size}"
                    f" by Node {node.name} {node.id} {node.rank_index}"
                )
                self._job_node_groups[node.group] = {}
            logger.debug(
                f"Update node group {node.group}/{node.group_id} with "
                f"Node {node.name} id {node.id} rank {node.rank_index}"
            )
            self._job_node_groups[node.group][node.rank_index] = copy.deepcopy(
                node
            )

    def clear_job_node_groups(self):
        with self._group_locker:
            self._job_node_groups = {}

    def report_failed_node(self, node_id: Optional[Union[int, str]] = None):
        if node_id is None:
            return

        node_id = int(node_id)
        with self._locker:
            if node_id not in self._failed_nodes:
                self._failed_nodes[node_id] = int(time.time())

    def get_job_restart_count(self):
        return self._job_restart_count

    def inc_job_restart_count(self):
        self._job_restart_count += 1
        return self._job_restart_count

    def get_failed_node_cnt(self):
        return len(self._failed_nodes)

    def set_pre_check_status(self, status: str):
        self._pre_check_status = status

    def get_pre_check_status(self):
        return self._pre_check_status

    def update_total_worker_num(self, worker_num: int):
        self._total_worker_num = worker_num

    def get_total_worker_num(self):
        return self._total_worker_num

    def request_stop(self, exit_code=-1, exit_reason=""):
        self._job_stage = JobStage.JOB_STOPPED
        if exit_code >= 0:
            self.set_exit_code(exit_code)
            self.set_exit_reason(exit_reason)

    def request_restart(self):
        self._job_stage = JobStage.JOB_RESTARTING

    def is_stopping(self):
        return self._job_stage == JobStage.JOB_STOPPING

    def is_stopped(self):
        return self._job_stage == JobStage.JOB_STOPPED

    def is_restarting(self):
        return self._job_stage == JobStage.JOB_RESTARTING

    def request_suspend(self):
        with self._locker:
            if (
                self._job_stage == JobStage.JOB_RUNNING
                or self._job_stage == JobStage.JOB_INIT
            ):
                logger.info("job is suspended")
                self._job_pre_status = self._job_stage
                self._job_stage = JobStage.JOB_SUSPENDED
            else:
                logger.info(f"{self._job_stage} job skip suspend")

    def request_unsuspend(self):
        with self._locker:
            if self._job_stage == JobStage.JOB_SUSPENDED:
                logger.info("job is unsuspended")
                self._job_stage = self._job_pre_status
            else:
                logger.info(f"{self._job_stage} job can not be unsuspend")

    def is_suspended(self):
        return self._job_stage == JobStage.JOB_SUSPENDED


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context
