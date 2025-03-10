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
from unittest.mock import patch

from dlrover.python.diagnosis.common.diagnosis_action import NoAction
from dlrover.python.diagnosis.diagnostician.diagnostician import (
    DiagnosticianManager,
)
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_diagnostician_mgr(self):
        mgr = DiagnosticianManager()

        diagnostician = FailureNodeDiagnostician()
        mgr.register_diagnostician("", diagnostician)
        self.assertEqual(len(mgr._diagnosticians), 0)

        name = "node_failed"
        action = mgr.observe(name)
        self.assertTrue(isinstance(action, NoAction))

        action = mgr.resolve(name, NoAction())
        self.assertTrue(isinstance(action, NoAction))

        mgr.register_diagnostician(name, diagnostician)
        action = mgr.resolve(name, NoAction())
        self.assertTrue(isinstance(action, NoAction))

    @patch(
        "dlrover.python.diagnosis.diagnostician"
        ".failure_node_diagnostician.FailureNodeDiagnostician.observe"
    )
    def test_failure_node_diagnostician(self, mock_diagnostician):
        mock_diagnostician.side_effect = Exception("mock_error")

        mgr = DiagnosticianManager()

        diagnostician = FailureNodeDiagnostician()
        mgr.register_diagnostician("failed_node", diagnostician)

        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 507035"

        action = mgr.observe("node_failed", log_file=file_path, errors=errors)
        self.assertTrue(isinstance(action, NoAction))
