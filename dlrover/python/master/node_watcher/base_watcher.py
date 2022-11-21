# Copyright 2022 The DLRover Authors. All rights reserved.
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
from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeResourceBoundary,
    NodeStatus,
)
from dlrover.python.common.resource import NodeResource


class Node(object):
    """Node records the information of each training node.
    Attributes:
        type: str, the type (e.g. "ps", "worker") of a node.
        id: int, the id of a node.
        name: str, the name of a node.
        status: the status of a node.
        start_time: int, the start timestamp of a node.
        task_index: int, the task index of a node in a training cluster.
        relaunch_count: int, the relaunched number of the training node.
        critical: bool, if true, the job will fail if the node fails.
        max_relaunch_count: int, the maximum to relaunch a node.
        relaunchable: bool, whether to relaunch a node if it fails.
        is_released: bool, whether to released the node.
        exit_reason: str, the exited reason of a node.
        used_resource: the resource usage of the node.
    """

    def __init__(
        self,
        node_type,
        node_id,
        name=None,
        status=NodeStatus.INITIAL,
        start_time=None,
        task_index=None,
        relaunch_count=0,
        critical=False,
        max_relaunch_count=0,
        relaunchable=True,
        service_addr=None,
    ):
        self.type = node_type
        self.id = node_id
        self.name = name
        self.status = status
        self.start_time = start_time
        self.task_index = task_index if task_index is not None else node_id
        self.relaunch_count = relaunch_count
        self.critical = critical
        self.max_relaunch_count = max_relaunch_count
        self.relaunchable = relaunchable
        self.service_addr = service_addr

        self.create_time = None
        self.finish_time = None
        self.is_recovered_oom = False
        self.is_released = False
        self.exit_reason = None
        self.used_resource = NodeResource(0.0, 0.0)

    def inc_relaunch_count(self):
        self.relaunch_count += 1

    def update_info(
        self,
        name=None,
        start_time=None,
        create_time=None,
    ):
        if name is not None:
            self.name = name
        if start_time is not None:
            self.start_time = start_time
        if create_time is not None:
            self.create_time = create_time

    def update_status(self, status=None):
        if status is not None:
            self.status = status

    def update_resource_usage(self, cpu, memory):
        self.used_resource.cpu = cpu
        self.used_resource.memory = memory

    def get_relaunch_node_info(self, new_id):
        new_node = copy.deepcopy(self)
        new_node.id = new_id
        new_node.name = None
        new_node.status = NodeStatus.INITIAL
        new_node.start_time = None
        new_node.is_released = False
        new_node.relaunchable = True
        return new_node

    def is_unrecoverable_failure(self):
        if (
            self.relaunch_count >= self.max_relaunch_count
            or self.exit_reason == NodeExitReason.FATAL_ERROR
            or self.used_resource.memory >= NodeResourceBoundary.MAX_MEMORY
        ):
            return True
        return False

    def set_exit_reason(self, reason):
        self.exit_reason = reason


class NodeEvent(object):
    """NodeEvent is the event to change the status of a Node"""

    def __init__(self, event_type, node):
        self.event_type = event_type
        self.node: Node = node


class NodeWatcher(metaclass=ABCMeta):
    def __init__(self, job_uuid):
        self._job_uuid = job_uuid

    @abstractmethod
    def watch(self):
        """Wath events of nodes and returns a generator"""
        pass

    @abstractmethod
    def list(self) -> List[Node]:
        """List all nodes of the job"""
        pass
