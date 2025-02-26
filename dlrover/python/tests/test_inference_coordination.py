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

import unittest

from dlrover.python.common.constants import EventReportConstants
from dlrover.python.diagnosis.common.constants import DiagnosisActionType
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)
from dlrover.python.diagnosis.inferencechain.coordinator import (
    coordinate_solutions,
)


class InferenceCoordinationTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_event_action(self):
        test_solutions = []
        self.assertEqual(
            coordinate_solutions(test_solutions).action_type,
            DiagnosisActionType.NONE,
        )

        test_solutions.append(
            Inference(
                name=InferenceName.ACTION,
                attribution=InferenceAttribute.IS,
                description=InferenceDescription.EVENT,
                configs={
                    "event_type": EventReportConstants.TYPE_WARN,
                    "event_instance": EventReportConstants.JOB_INSTANCE,
                    "event_action": EventReportConstants.ACTION_HANG_WARN,
                    "event_msg": "",
                    "event_labels": "{}",
                },
            )
        )
        action = coordinate_solutions(test_solutions)
        self.assertEqual(action.action_type, DiagnosisActionType.EVENT)
        self.assertEqual(action.event_type, EventReportConstants.TYPE_WARN)
        self.assertEqual(
            action.event_instance, EventReportConstants.JOB_INSTANCE
        )
        self.assertEqual(
            action.event_action, EventReportConstants.ACTION_HANG_WARN
        )
        self.assertEqual(action.event_msg, "")
        self.assertEqual(action.event_labels, {})


if __name__ == "__main__":
    unittest.main()
