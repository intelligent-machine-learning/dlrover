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

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisErrorConstant,
    DiagnosticianType,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.diagnosis.datacollector.data_collector import (
    SimpleDataCollector,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_failure import (  # noqa: E501
    ResourceCollectionFailureDiagnostician,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.util.function_util import TimeoutException


class DiagnosisManagerTest(unittest.TestCase):
    def setUp(self):
        self._agent_context = get_agent_context()

    def tearDown(self):
        self._agent_context.clear_action_queue()

    def test_diagnosis_mgr(self):
        context = get_agent_context()
        mgr = DiagnosisManager(context)

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

        # test register diagnosis
        mgr.register_diagnostician("unknown", Diagnostician(), 60)
        self.assertEqual(len(mgr._diagnosticians), 2)

        mgr.register_diagnostician(
            name, Diagnostician(), DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL - 5
        )
        self.assertEqual(
            mgr._diagnosticians[name][1],
            DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL,
        )

        # test start diagnosis
        mgr.start_diagnosis()
        thread_name = f"periodical_diagnose_{name}"
        thread_names = [t.name for t in threading.enumerate()]
        self.assertIn(thread_name, thread_names, f"Not found {thread_name}")

        # test register_periodical_collector
        collector = SimpleDataCollector()
        mgr.register_periodical_data_collector(collector, 1)
        self.assertEqual(
            mgr._periodical_collector[collector],
            DiagnosisManager.MIN_DATA_COLLECT_INTERVAL,
        )

        # test start data collector
        mgr.start_data_collection()
        thread_name = f"periodical_collector_{collector.__class__.__name__}"
        thread_names = [t.name for t in threading.enumerate()]
        self.assertIn(thread_name, thread_names, f"Not found {thread_name}")

    def test_diagnosis_mgr_exception(self):
        context = get_agent_context()
        mgr = DiagnosisManager(context)

        name = "test"
        diagnostician = Diagnostician()
        mgr.register_diagnostician(name, diagnostician)

        with self.assertLogs(logger, level="ERROR") as log_capture:
            # test observe exception
            diagnostician.observe = MagicMock(side_effect=TimeoutException())
            mgr.observe(name)
            err_msg = f"{diagnostician.__class__.__name__}.observe is timeout"
            self.assertTrue(
                any(err_msg in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            diagnostician.observe = MagicMock(side_effect=Exception())
            ob = mgr.observe(name)
            self.assertTrue(
                any("Fail to observe" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            # test resolve exception
            diagnostician.resolve = MagicMock(side_effect=TimeoutException())
            mgr.resolve(name, ob)
            err_msg = f"{diagnostician.__class__.__name__}.resolve is timeout"
            self.assertTrue(
                any(err_msg in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            diagnostician.resolve = MagicMock(side_effect=Exception())
            mgr.resolve(name, ob)
            self.assertTrue(
                any("Fail to resolve" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

    @patch(
        "dlrover.python.diagnosis.common.diagnostician.Diagnostician.diagnose"
    )
    def test_start_periodical_diagnosis(self, mock_diagnose):
        context = get_agent_context()
        mgr = DiagnosisManager(context)
        diagnostician = Diagnostician()
        name = "test"
        # use 0.1 for testing
        mgr._diagnosticians[name] = (diagnostician, 0.1)

        with self.assertLogs(logger, level="ERROR") as log_capture:
            thread = threading.Thread(
                target=mgr._start_periodical_diagnosticians,
                name="diagnosis_thread",
                args=(name,),
                daemon=True,
            )
            thread.start()
            time.sleep(0.2)
            self.assertTrue(context._diagnosis_action_queue.len() > 0)

            mock_diagnose.side_effect = Exception()
            time.sleep(0.2)
            self.assertTrue(
                any("Fail to diagnose" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

    @patch(
        "dlrover.python.diagnosis.datacollector"
        ".data_collector.SimpleDataCollector.collect_data"
    )
    def test_start_periodical_collector(self, mock_collect):
        context = get_agent_context()
        mgr = DiagnosisManager(context)

        collector = SimpleDataCollector()

        diagnostician = ResourceCollectionFailureDiagnostician()
        mgr.register_diagnostician(
            DiagnosticianType.RESOURCE_COLLECT_FAILURE, diagnostician
        )

        with self.assertLogs(logger, level="ERROR") as log_capture:
            thread = threading.Thread(
                target=mgr._start_periodical_collector,
                name="collect_thread",
                args=(
                    collector,
                    0.1,
                ),
                daemon=True,
            )
            thread.start()

            mock_collect.side_effect = TimeoutException()
            time.sleep(0.2)
            self.assertTrue(
                any("timeout" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            mock_collect.side_effect = Exception(
                DiagnosisErrorConstant.GPU_LOST
            )
            time.sleep(0.2)
            self.assertTrue(context._diagnosis_action_queue.len() > 0)
