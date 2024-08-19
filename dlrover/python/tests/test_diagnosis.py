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

from dlrover.python.diagnosis.common.diagnosis_data import CudaLog
from dlrover.python.master.diagnosis.diagnosis import DiagnosisDataManager


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DiagnosisDataManager(5)
        data_type = "type"
        log1 = CudaLog(0)
        mgr.store_data(data_type, log1)
        time.sleep(1)
        log2 = CudaLog(0)
        mgr.store_data(data_type, log2)

        logs = mgr.get_data(data_type)
        self.assertEqual(len(logs), 2)

        time.sleep(6)
        log3 = CudaLog(0)
        mgr.store_data(data_type, log3)
        logs = mgr.get_data(data_type)
        self.assertEqual(len(logs), 1)


if __name__ == "__main__":
    unittest.main()
