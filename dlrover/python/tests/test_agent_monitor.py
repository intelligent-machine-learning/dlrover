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

from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor
from dlrover.python.elastic_agent.monitor.training import (
    TrainingProcessReporter,
    is_tf_chief,
)


class ResourceMonitorTest(unittest.TestCase):
    def test_resource_monitor(self):
        resource_monitor = ResourceMonitor()
        time.sleep(0.3)
        resource_monitor.report_resource()
        self.assertTrue(resource_monitor._total_cpu >= 0.0)

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
        reporter0 = TrainingProcessReporter()
        reporter1 = TrainingProcessReporter()
        self.assertEqual(reporter0, reporter1)


if __name__ == "__main__":
    unittest.main()
