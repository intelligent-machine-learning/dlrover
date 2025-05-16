# Copyright 2025 The DLRover Authors. All rights reserved.
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

import os
import unittest
from unittest.mock import MagicMock

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisErrorConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_error_diagnostician import (  # noqa: E501
    ResourceCollectErrorDiagnostician,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.util.function_util import TimeoutException


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        self._master, self._addr = start_local_master()
        MasterClient._instance = build_master_client(self._addr, 1)

    def tearDown(self):
        os.environ.clear()
        self._master.stop()

    def test_diagnostician(self):
        diagnostician = Diagnostician()

        ob = diagnostician.observe()
        self.assertEqual(ob.observation, "unknown")

        action = diagnostician.resolve(ob)
        self.assertTrue(isinstance(action, EventAction))

        action = diagnostician.diagnose()
        self.assertTrue(isinstance(action, EventAction))

        diagnostician.resolve = MagicMock(side_effect=Exception())
        action = diagnostician.diagnose()
        self.assertTrue(isinstance(action, NoAction))

        with self.assertLogs(logger, level="ERROR") as log_capture:
            diagnostician.observe = MagicMock(side_effect=TimeoutException())
            diagnostician.diagnose()
            self.assertTrue(
                any("timeout" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

    def test_failure_node_diagnostician(self):
        diagnostician = FailureNodeDiagnostician()

        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 507035"

        ob = diagnostician.observe(log_file=file_path, errors=errors)
        self.assertEqual(ob.observation, DiagnosisErrorConstant.NODE_FAILED)

        ob = diagnostician.observe(log_file=file_path)
        self.assertTrue(len(ob.observation) == 0)

        ob = diagnostician.observe(errors=errors)
        self.assertTrue(len(ob.observation) == 0)

    def test_resource_collect_error_diagnostician(self):
        error_log = "GPU is lost"

        diagnostician = ResourceCollectErrorDiagnostician()

        action = diagnostician.diagnose(error_log=error_log)
        self.assertTrue(isinstance(action, EventAction))
