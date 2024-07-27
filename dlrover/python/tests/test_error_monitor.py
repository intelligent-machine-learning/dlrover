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

from dlrover.python.common.constants import NodeType, TrainingExceptionLevel
from dlrover.python.common.node import Node
from dlrover.python.master.monitor.error_monitor import SimpleErrorMonitor


class ErrorLogMonitorTest(unittest.TestCase):
    def test_process_error(self):
        err_monitor = SimpleErrorMonitor()
        node = Node(NodeType.WORKER, 0)
        error_data = "RuntimeError"
        relaunched = err_monitor.process_error(
            node=node,
            restart_count=1,
            error_data=error_data,
            level=TrainingExceptionLevel.PROCESS_ERROR,
        )
        self.assertFalse(relaunched)
        error_data = "The node is down."
        relaunched = err_monitor.process_error(
            node=node,
            restart_count=1,
            error_data=error_data,
            level=TrainingExceptionLevel.NODE_ERROR,
        )
        self.assertTrue(relaunched)

        err_monitor.process_error(
            node=None,
            restart_count=0,
            error_data="test",
            level=TrainingExceptionLevel.ERROR,
        )
