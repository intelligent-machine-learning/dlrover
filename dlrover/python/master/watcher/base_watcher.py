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

from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.node import Node


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
