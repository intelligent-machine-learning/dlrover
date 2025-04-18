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

from dlrover.python.diagnosis.common.constants import DiagnosisErrorConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_error_diagnostician import (  # noqa: E501
    ResourceCollectErrorDiagnostician,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        self._master, self._addr = start_local_master()
        MasterClient._instance = build_master_client(self._addr, 1)

    def tearDown(self):
        os.environ.clear()
        self._master.stop()

    @patch(
        "dlrover.python.diagnosis.common"
        ".diagnostician.Diagnostician.observe"
    )
    @patch(
        "dlrover.python.diagnosis.common"
        ".diagnostician.Diagnostician.resolve"
    )
    def test_diagnostician_exception(self, mock_resolve, mock_observe):
        mock_observe.side_effect = Exception("observe_error")
        mock_resolve.side_effect = Exception("resolve_error")

        context = get_agent_context()
        mgr = DiagnosisManager(context)

        name = "test"
        diagnostician = Diagnostician()
        mgr.register_diagnostician(name, diagnostician)

        ob = mgr.observe(name)
        self.assertTrue(len(ob.observation) == 0)

        action = mgr.resolve(name, ob)
        self.assertTrue(isinstance(action, NoAction))

        action = mgr.diagnose(name)
        self.assertTrue(isinstance(action, NoAction))

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
