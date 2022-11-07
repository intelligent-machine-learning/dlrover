# Copyright 2022 The EasyDL Authors. All rights reserved.
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

import unittest

from dlrover.python.common.resource import NodeResource
from dlrover.python.master.stats_collector.local_collector import (
    LocalStatsCollector,
)


class ALocalStatsCollectorTest(unittest.TestCase):
    def test_report_resource_usage(self):
        collector = LocalStatsCollector("123")
        resource = NodeResource(cpu=1, memory=1024)
        worker0_name = "worker-0"
        worker1_name = "worker-1"
        collector.collect_node_resource_usage(worker0_name, resource)
        collector.report_resource_usage()
        collector.collect_node_resource_usage(worker0_name, resource)
        collector.collect_node_resource_usage(worker1_name, resource)
        collector.report_resource_usage()
        self.assertEqual(len(collector._all_node_resources), 2)
        self.assertEqual(len(collector._all_node_resources[worker0_name]), 2)
        self.assertEqual(len(collector._all_node_resources[worker1_name]), 1)
