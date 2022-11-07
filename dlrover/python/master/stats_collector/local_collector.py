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

from dlrover.python.master.stats_collector.base_collector import StatsCollector


class LocalStatsCollector(StatsCollector):
    def __init__(self, job_uuid):
        super(LocalStatsCollector, self).__init__(job_uuid)
        self._all_node_resources = {}

    def report_resource_usage(self):
        for node_name, resource in self._node_resource_usage.items():
            self._all_node_resources.setdefault(node_name, [])
            self._all_node_resources[node_name].append(resource)
