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

from dlrover.python.common.constants import NodeType
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor


class SpeedMonitorTest(unittest.TestCase):
    def test_speed_monitor(self):
        monitor = SpeedMonitor()
        monitor.set_target_worker_num(2)
        monitor.add_running_worker(NodeType.WORKER, 0)
        monitor.add_running_worker(NodeType.WORKER, 1)
        monitor.collect_global_step(1, 1)
        monitor.collect_global_step(301, 11),
        monitor.collect_global_step(9001, 301),
        self.assertEqual(monitor.completed_global_step, 9001)
        self.assertTrue(monitor.worker_adjustment_finished())
        self.assertEqual(monitor.running_speed, 30)
        monitor.remove_running_worker(NodeType.WORKER, 1)
        monitor.collect_global_step(18001, 601)
        self.assertFalse(monitor.worker_adjustment_finished())
