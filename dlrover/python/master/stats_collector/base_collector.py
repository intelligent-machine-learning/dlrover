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


class StatsCollector(metaclass=ABCMeta):
    def __init__(self, job_uuid):
        self._job_uuid = job_uuid
        self._node_resource_usage = {}

    def collect_node_resource_usage(self, node_name, resource):
        """Collect the resource usage of the node_name node.
        Args:
            node_name: string, the name of node
            resource: elasticl.python.resource.NodeResource instace,
                the resource usage of the node.
        """
        self._node_resource_usage[node_name] = resource

    @abstractmethod
    def report_resource_usage(self):
        pass
