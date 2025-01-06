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
from typing import List
from unittest import mock

from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisDataType,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    DiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    is_training_hanged,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.diagnosis.diagnosis_manager import DiagnosisManager


class DiagnosisManagerTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DiagnosisDataManager(1)
        log1 = TrainingLog(0)
        mgr.store_data(log1)
        time.sleep(0.01)
        log2 = TrainingLog(0)
        mgr.store_data(log2)

        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 2)

        time.sleep(1.5)
        log3 = TrainingLog(0)
        mgr.store_data(log3)
        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 1)

    def test_diagnosis_manager_api(self):
        mgr = DiagnosisManager()
        mgr.pre_check()
        mgr.start_observing()
        mgr.stop_observing()

    def test_diagnosis_manager(self):
        mgr = DiagnosisManager()
        problems: List[Inference] = [
            Inference(
                InferenceName.TRAINING,
                InferenceAttribute.ISORNOT,
                InferenceDescription.HANG,
            )
        ]
        mgr._diagnostician.register_training_problems(problems)
        self.assertEqual(len(mgr._diagnostician._training_problems), 1)

        data_mgr = DiagnosisDataManager(10000)
        operator = CheckTrainingHangOperator(data_mgr)
        mgr._diagnostician.register_observers([operator])
        self.assertEqual(len(mgr._diagnostician._observers), 1)

        data = DiagnosisData(
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content="XPU_TIMER_COMMON_HANG",
        )
        data_mgr.store_data(data)

        # mock training hang
        mgr._diagnostician._observers[0].is_hang = mock.MagicMock(
            return_value=True
        )

        # observe training problems
        observed_problems = mgr._diagnostician.observe_training()
        self.assertTrue(is_training_hanged(observed_problems[0]))

        # explore solutions to observed problems
        action = mgr._diagnostician.resolve_problems(observed_problems)
        self.assertEqual(action.action_type, DiagnosisActionType.NONE)
