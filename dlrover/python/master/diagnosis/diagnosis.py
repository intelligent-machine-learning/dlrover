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
from dlrover.python.master.diagnosis.diagnosis_data import (
    DataManager,
    DiagnosisData,
)
from dlrover.python.master.diagnosis.diagnostician import Diagnostician
from dlrover.python.master.diagnosis.inferencechain.common import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)


class DiagnosisManager:
    def __init__(self):
        self.data_manager: DataManager = DataManager(600)
        self.diagnostician: Diagnostician = Diagnostician(self.data_manager)

    def collect_diagnosis_data(self, data_type: str, data: DiagnosisData):
        self.data_manager.store_data(data_type, data)

    def start(self):
        logger.info("Start Diagnosis Manager ...")
        problems: list[Inference] = [
            Inference(
                InferenceName.TRAINING,
                InferenceAttribute.ISORNOT,
                InferenceDescription.HANG,
            )
        ]
        self.diagnostician.register_problems(problems)

        try:
            thread = threading.Thread(
                target=self._diagnose_failures(),
                name="diagnose_failures",
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

    def _diagnose_failures(self):
        logger.info("Start to diagnose failures")
        while True:
            observed_problems = self.diagnostician.observe_training()
            for problem in observed_problems:
                logger.info(f"observed problems: {problem}")
                root_causes = self.diagnostician.diagnose_failure(problem)
                for root_cause in root_causes:
                    logger.info(f"identify root cause: {root_cause}")
            time.sleep(180)
