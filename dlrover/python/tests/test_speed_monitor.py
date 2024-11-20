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

import time
import unittest

from dlrover.python.common.constants import NodeType
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor


class SpeedMonitorTest(unittest.TestCase):
    def test_monitor_running_workers(self):
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

    def test_monitor_eval_time(self):
        monitor = SpeedMonitor()
        monitor.set_worker_start_eval_time(0)
        time.sleep(0.1)
        monitor.update_worker_eval_time(0)
        eval_time = monitor.get_worker_eval_time(0)
        self.assertTrue(eval_time > 0.1)

    def test_monitor_user_step(self):
        monitor = SpeedMonitor()
        ts = time.time()
        monitor.max_step_count = 3
        monitor.collect_user_step(ts, 1, 1000)
        self.assertEqual(monitor.first_step_time, ts)
        monitor.collect_user_step(time.time(), 2, 1000)
        monitor.collect_user_step(time.time(), 3, 1000)
        self.assertEqual(len(monitor.user_step_records), 3)
        self.assertEqual(monitor.user_step_records[0].step_num, 1)
        self.assertEqual(monitor.user_step_records[0].total_step, 1000)
        self.assertEqual(monitor.user_step_records[0].timestamp, ts)
        self.assertEqual(monitor.user_step_records[1].step_num, 2)
        self.assertEqual(monitor.user_step_records[2].step_num, 3)
        monitor.collect_user_step(time.time(), 4, 1000)
        self.assertEqual(len(monitor.user_step_records), 3)
        self.assertEqual(monitor.user_step_records[0].step_num, 2)
        monitor.collect_user_step(time.time(), 10, 1000)
        self.assertEqual(len(monitor.user_step_records), 1)
