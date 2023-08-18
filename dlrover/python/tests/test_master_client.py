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

import unittest 

from dlrover.python.elastic_agent.master_client import build_master_client
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.common.constants import TrainingMsgLevel, NodeType


class MasterClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        self._master_client = build_master_client(addr)

    def addCleanup(self):
        self._master.stop()

    def test_report_used_resource(self):
        self._master_client.report_used_resource(1024, 10)
        node = self._master.job_manager._job_nodes[NodeType.WORKER][0]
        self.assertEqual(node.used_resource.cpu, 10)
        self.assertEqual(node.used_resource.memory, 1024)

    def test_report_failures(self):
        res = self._master_client.report_failures(
            "test", 0, TrainingMsgLevel.WARNING
        )
        self.assertIsNone(res)


