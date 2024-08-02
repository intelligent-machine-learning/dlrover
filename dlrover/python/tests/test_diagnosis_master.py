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

from dlrover.python.common.diagnosis import CudaLog, DiagnosisDataType
from dlrover.python.master.diagnosis.diagnosis_data import DataManager
from dlrover.python.master.diagnosis.diagnostician import Diagnostician
from dlrover.python.master.diagnosis.inferencechain.common import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    same_inference,
)
from dlrover.python.master.diagnosis.operator.check_training_hang_operator import CheckTrainingHangOperator


def create_data_mgr() -> DataManager:
    mgr = DataManager(10000)

    trace1 = "MainThread;func3@file3;func1@file1;func2@file2"
    trace2 = "MainThread;wait@wait.cc;func1@file1;func2@file2"
    world_size = 4
    cuda_log11 = CudaLog(0, world_size, {trace1: {0, 1}})
    cuda_log12 = CudaLog(0, world_size, {trace2: {0, 1}})
    cuda_log21 = CudaLog(0, world_size, {trace1: {2, 3}})
    cuda_log22 = CudaLog(0, world_size, {trace2: {2, 3}})

    mgr.store_data(0, DiagnosisDataType.CUDALOG, cuda_log11)
    mgr.store_data(1, DiagnosisDataType.CUDALOG, cuda_log21)
    for i in range(0, 3):
        mgr.store_data(0, DiagnosisDataType.CUDALOG, cuda_log12)
        mgr.store_data(1, DiagnosisDataType.CUDALOG, cuda_log22)
    return mgr


class MasterDiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DataManager(5)
        mgr.store_data(0, DiagnosisDataType.CUDALOG, CudaLog(0, 0, {}))
        time.sleep(2)
        mgr.store_data(0, DiagnosisDataType.CUDALOG, CudaLog(0, 0, {}))
        mgr.store_data(1, DiagnosisDataType.CUDALOG, CudaLog(0, 0, {}))

        nodes_cuda_logs = mgr.get_nodes_cuda_logs()
        self.assertEqual(len(nodes_cuda_logs[0]), 2)
        self.assertEqual(len(nodes_cuda_logs[1]), 1)

        time.sleep(4)
        mgr.store_data(0, DiagnosisDataType.CUDALOG, CudaLog(0, 0, {}))
        nodes_cuda_logs = mgr.get_nodes_cuda_logs()
        self.assertEqual(len(nodes_cuda_logs[0]), 2)

    def test_diagnostician(self):
        data_mgr = create_data_mgr()
        diagnostician = Diagnostician(data_mgr)
        problems: list[Inference] = [
            Inference(
                name=InferenceName.TRAINING,
                attribution=InferenceAttribute.ISORNOT,
                description=InferenceDescription.HANG,
            )
        ]
        diagnostician.register_problems(problems)

        infs = diagnostician.observe_training()
        self.assertEqual(len(infs), 1)

        inf = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.HANG,
        )
        self.assertTrue(same_inference(infs[0], inf))

    def test_training_hang_operator(self):
        mgr = create_data_mgr()

        operator = CheckTrainingHangOperator(mgr)
        problem = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.HANG,
        )
        self.assertTrue(operator.is_compatible(problem))

        infs = operator.infer([])
        self.assertEqual(len(infs), 1)

        inf = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.HANG,
        )
        self.assertTrue(same_inference(infs[0], inf))


if __name__ == "__main__":
    unittest.main()
