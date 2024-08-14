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

import json
import os
import time
import unittest
from unittest.mock import patch

from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.grpc import GPUStats
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor
from dlrover.python.elastic_agent.monitor.training import (
    TFTrainingReporter,
    is_tf_chief,
)
from dlrover.python.tests.test_utils import start_local_master


class ResourceMonitorTest(unittest.TestCase):
    def setUp(self):
        self.master_proc, self.addr = start_local_master()
        MasterClient._instance = build_master_client(self.addr, 0.5)

    def tearDown(self):
        self.master_proc.stop()

    def test_resource_monitor(self):
        gpu_stats: list[GPUStats] = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]
        mock_env = {
            NodeEnv.DLROVER_MASTER_ADDR: self.addr,
            NodeEnv.MONITOR_ENABLED: "true",
        }

        with patch.dict("os.environ", mock_env):
            result = not os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "") or not (
                os.getenv(NodeEnv.MONITOR_ENABLED, "") == "true"
            )
            self.assertFalse(result)
            # mock get_gpu_stats
            with patch(
                "dlrover.python.elastic_agent.monitor.resource.get_gpu_stats",
                return_value=gpu_stats,
            ):
                with patch("pynvml.nvmlInit"):
                    resource_monitor = ResourceMonitor.singleton_instance()
                    resource_monitor.start()
                    time.sleep(0.3)
                    resource_monitor.report_resource()
                    self.assertTrue(resource_monitor._total_cpu >= 0.0)
                    self.assertTrue(resource_monitor._gpu_stats == gpu_stats)

    def test_training_reporter(self):
        TF_CONFIG = {
            "cluster": {
                "chief": ["localhost:2221"],
                "worker": ["localhost:2222"],
                "ps": ["localhost:2226"],
            },
            "task": {"type": "chief", "index": 0},
        }
        os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)
        self.assertTrue(is_tf_chief())
        reporter0 = TFTrainingReporter.singleton_instance()
        reporter1 = TFTrainingReporter.singleton_instance()
        self.assertEqual(reporter0, reporter1)
        reporter0.set_start_time()
        self.assertTrue(reporter0._start_time > 0)
        reporter0._last_timestamp = time.time() - 30
        reporter0.report_resource_with_step(100)


if __name__ == "__main__":
    unittest.main()
