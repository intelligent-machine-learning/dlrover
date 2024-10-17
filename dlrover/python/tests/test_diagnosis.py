# Copyright 2024 The DLRover Authors. All rights reserved.
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

from dlrover.python.common.constants import NodeStatus
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisDataType,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NodeRelaunchAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import TrainingLog
from dlrover.python.master.diagnosis.diagnosis import DiagnosisDataManager


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DiagnosisDataManager(expire_time_period=3)
        log1 = TrainingLog(0)
        mgr.store_data(log1)
        time.sleep(1)
        log2 = TrainingLog(0)
        mgr.store_data(log2)

        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 2)

        time.sleep(4)
        log3 = TrainingLog(0)
        mgr.store_data(log3)
        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 1)

    def test_action_basic(self):
        basic_action = DiagnosisAction()
        self.assertEqual(
            basic_action.diagnosis_type, DiagnosisActionType.NO_ACTION
        )

        event_action = EventAction(
            "info", "job", "test", "test123", {"k1": "v1"}
        )
        self.assertEqual(
            event_action.diagnosis_type, DiagnosisActionType.EVENT
        )
        self.assertEqual(event_action.event_type, "info")
        self.assertEqual(event_action.instance, "job")
        self.assertEqual(event_action.action, "test")
        self.assertEqual(event_action.msg, "test123")
        self.assertEqual(event_action.labels, {"k1": "v1"})

        node_relaunch_action = NodeRelaunchAction(1, NodeStatus.FAILED, "hang")
        self.assertEqual(
            node_relaunch_action.diagnosis_type,
            DiagnosisActionType.MASTER_RELAUNCH_WORKER,
        )
        self.assertEqual(node_relaunch_action.node_id, 1)
        self.assertEqual(node_relaunch_action.node_status, NodeStatus.FAILED)
        self.assertEqual(node_relaunch_action.reason, "hang")


if __name__ == "__main__":
    unittest.main()
