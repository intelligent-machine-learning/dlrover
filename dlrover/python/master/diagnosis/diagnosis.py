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

import threading
import time

from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.diagnosis.analyst import Analyst
from dlrover.python.master.diagnosis.diagnosis_data import (
    DataManager,
    DiagnosisData,
)
from dlrover.python.master.diagnosis.diagnostician import Diagnostician


class DiagnosisManager:
    def __init__(self):
        self.data_manager: DataManager = DataManager(600)
        self.analyst: Analyst = Analyst(self.data_manager)
        self.diagnostician: Diagnostician = Diagnostician()

    def collect_diagnosis_data(self, data_type: str, data: DiagnosisData):
        self.data_manager.store_data(data_type, data)

    def start(self):
        logger.info("Start Diagnosis Manager ...")

        try:
            thread = threading.Thread(
                target=self._diagnose_faults(),
                name="diagnosis_faults",
                daemon=True,
            )
            thread.start()
            if thread.is_alive():
                logger.info("Diagnosis Manager is started")
        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis manager thread. Error: {e}"
            )

    def stop(self):
        pass

    def _diagnose_faults(self):
        logger.info("Start to diagnose faults")
        while True:
            observed_problems = self.analyst.observe_training()
            for problem in observed_problems:
                logger.info(problem.to_string())
            time.sleep(180)
