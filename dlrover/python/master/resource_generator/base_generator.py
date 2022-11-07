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


class ResourcePlan(object):
    """A resource configuration plan."""

    def __init__(self):
        self.task_group_resources = {}
        self.node_resources = {}

    def add_task_group_resource(self, name, resource):
        """Add task group resource.
        Args:
            name: string, the name of task group like "ps/worker".
            resource: a dlrover.python.common.resource.TaskGroupResource
                instance.
        """
        self.task_group_resources[name] = resource

    def add_node_resource(self, name, resource):
        """Add a node resource.
        Args:
            name: string, the name of node.
            resource: a dlrover.python.common.resource.NodeResource
                instance.
        """
        self.node_resources[name] = resource


class ResourceGenerator(metaclass=ABCMeta):
    def __init__(self, job_uuid):
        self._job_uuid = job_uuid

    @abstractmethod
    def generate_plan(self, stage, config={}):
        """Generate a resource configuration p"""
        pass
