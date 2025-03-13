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
import threading
import time
import unittest
from unittest.mock import patch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import Observation
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.diagnostician.diagnostician import (
    Diagnostician,
    DiagnosticianManager,
)
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.util.function_util import TimeoutException


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_diagnostician_mgr(self):
        context = get_agent_context()
        mgr = DiagnosticianManager(context)

        # Test basic function
        diagnostician = Diagnostician()
        mgr.register_diagnostician("", diagnostician)
        self.assertEqual(len(mgr._diagnosticians), 0)

        name = "test"
        ob = mgr.observe(name)
        self.assertTrue(len(ob.observation) == 0)

        action = mgr.resolve(name, ob)
        self.assertTrue(isinstance(action, NoAction))

        action = mgr.diagnose(name)
        self.assertTrue(isinstance(action, NoAction))

        mgr.register_diagnostician(name, diagnostician)
        ob = mgr.observe(name)
        self.assertTrue(len(ob.observation) > 0)

        action = mgr.resolve(name, ob)
        self.assertTrue(isinstance(action, EventAction))

        action = mgr.diagnose(name)
        self.assertTrue((isinstance(action, EventAction)))

        # test register_periodical_diagnosis
        mgr.register_periodical_diagnosis("unknown", 60)
        self.assertTrue(len(mgr._periodical_diagnosis) == 0)

        mgr.register_periodical_diagnosis(
            name, DiagnosticianManager.MIN_DIAGNOSIS_INTERVAL - 5
        )
        self.assertEqual(
            mgr._periodical_diagnosis[name],
            DiagnosticianManager.MIN_DIAGNOSIS_INTERVAL,
        )

        # test start
        mgr.start()
        thread_name = f"periodical_diagnose_{name}"
        thread_names = [t.name for t in threading.enumerate()]
        self.assertIn(thread_name, thread_names, f"Not found {thread_name}")

    @patch(
        "dlrover.python.diagnosis.diagnostician"
        ".diagnostician.Diagnostician.observe"
    )
    @patch(
        "dlrover.python.diagnosis.diagnostician"
        ".diagnostician.Diagnostician.resolve"
    )
    def test_diagnostician_observe_exception(self, mock_resolve, mock_observe):
        mock_observe.side_effect = Exception("observe_error")
        mock_resolve.side_effect = Exception("resolve_error")

        context = get_agent_context()
        mgr = DiagnosticianManager(context)

        name = "test"
        diagnostician = Diagnostician()
        mgr.register_diagnostician(name, diagnostician)

        ob = mgr.observe(name)
        self.assertTrue(len(ob.observation) == 0)

        action = mgr.resolve(name, ob)
        self.assertTrue(isinstance(action, NoAction))

        action = mgr.diagnose(name)
        self.assertTrue(isinstance(action, NoAction))

    @patch(
        "dlrover.python.diagnosis.diagnostician"
        ".diagnostician.Diagnostician.diagnose"
    )
    def test_diagnostician_start_periodical_diagnosis(self, mock_diagnose):
        context = get_agent_context()
        mgr = DiagnosticianManager(context)
        diagnostician = Diagnostician()
        name = "test"
        mgr.register_diagnostician(name, diagnostician)
        mgr._periodical_diagnosis[name] = 0.1

        with self.assertLogs(logger, level="ERROR") as log_capture:
            thread = threading.Thread(
                target=mgr._start_periodical_diagnosis,
                name="test",
                args=(name,),
                daemon=True,
            )
            thread.start()
            time.sleep(0.2)
            self.assertTrue(context._diagnosis_action_queue.len() > 0)

            mock_diagnose.side_effect = TimeoutException()
            time.sleep(0.2)
            self.assertTrue(
                any("timeout" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            mock_diagnose.side_effect = Exception()
            time.sleep(0.2)
            self.assertTrue(
                any("Fail to diagnose" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

    def test_failure_node_diagnostician(self):
        diagnostician = FailureNodeDiagnostician()

        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 507035"

        ob = diagnostician.observe(log_file=file_path, errors=errors)
        self.assertEqual(ob.observation, Observation.NODE_FAILED)

        ob = diagnostician.observe(log_file=file_path)
        self.assertTrue(len(ob.observation) == 0)

        ob = diagnostician.observe(errors=errors)
        self.assertTrue(len(ob.observation) == 0)
