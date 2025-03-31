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
import unittest

from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.elastic_agent.context import get_agent_context


class DiagnosisManagerTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

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

        # test register_periodical_diagnosis
        mgr.register_periodical_diagnosis("unknown", 60)
        self.assertTrue(len(mgr._periodical_diagnosis) == 0)

        mgr.register_periodical_diagnosis(
            name, DiagnosisManager.MIN_DIAGNOSIS_INTERVAL - 5
        )
        self.assertEqual(
            mgr._periodical_diagnosis[name],
            DiagnosisManager.MIN_DIAGNOSIS_INTERVAL,
        )

        # test start
        mgr.start_diagnosis()
        thread_name = f"periodical_diagnose_{name}"
        thread_names = [t.name for t in threading.enumerate()]
        self.assertIn(thread_name, thread_names, f"Not found {thread_name}")
