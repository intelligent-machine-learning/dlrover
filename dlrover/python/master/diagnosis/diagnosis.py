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
from datetime import datetime, timedelta
from typing import Dict, List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.inference_chain import (
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    Inference,
    InferenceChain,
    InferenceOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)


def has_expired(timestamp: float, time_period: int) -> bool:
    dt = datetime.fromtimestamp(timestamp)
    expired_dt = dt + timedelta(seconds=time_period)
    return expired_dt < datetime.now()


class DiagnosisManager:
    def __init__(self):
        self._is_observing_started = False
        self._data_manager: DiagnosisDataManager = DiagnosisDataManager(600)
        self._diagnostician: Diagnostician = Diagnostician(self._data_manager)

    def collect_diagnosis_data(self, data_type: str, data: DiagnosisData):
        self._data_manager.store_data(data_type, data)

    def pre_check(self):
        logger.info("Start Diagnosis Manager to pre-check training...")

        # TODO
        pre_checks = []
        self._diagnostician.register_pre_check(pre_checks)
        pass

    def start_observing(self):
        logger.info("Start Diagnosis Manager to observing training...")
        self._is_observing_started = True

        problems: List[Inference] = [
            Inference(
                InferenceName.TRAINING,
                InferenceAttribute.ISORNOT,
                InferenceDescription.HANG,
            )
        ]
        self._diagnostician.register_problems(problems)

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

    def stop_observing(self):
        logger.info("Stop Diagnosis Manager to observing training.")
        self._is_observing_started = False

    def _diagnose_failures(self):
        logger.info("Start to diagnose failures for observing.")
        while True:
            if not self._is_observing_started:
                logger.info("Stop to diagnose failures for observing.")
                break

            observed_problems = self._diagnostician.observe_training()
            for problem in observed_problems:
                logger.info(f"observed problems: {problem}")
                root_causes = self._diagnostician.diagnose_failure(problem)
                for root_cause in root_causes:
                    logger.info(f"identify root cause: {root_cause}")
            time.sleep(180)


class DiagnosisDataManager:
    def __init__(self, expire_time_period):
        self.diagnosis_data: Dict[str, List[DiagnosisData]] = {}
        self.expire_time_period = expire_time_period

    def store_data(self, data_type: str, data: DiagnosisData):
        if data_type not in self.diagnosis_data:
            logger.warning(f"{data_type} is not found in the store")
            self.diagnosis_data[data_type] = []
        self.diagnosis_data[data_type].append(data)
        self._clean_diagnosis_data(data_type)

    def get_data(self, data_type: str) -> List[DiagnosisData]:
        if data_type not in self.diagnosis_data:
            return []
        return self.diagnosis_data[data_type]

    def _clean_diagnosis_data(self, data_type: str):
        if data_type not in self.diagnosis_data:
            return

        data = self.diagnosis_data[data_type]
        n = 0
        for d in data:
            if has_expired(d.get_timestamp(), self.expire_time_period):
                n = n + 1
            else:
                break

        self.diagnosis_data[data_type] = data[n:]


class Diagnostician:
    def __init__(self, data_manager):
        self._data_manager = data_manager
        self._pre_checks: List[Inference] = []
        self._training_problems: List[Inference] = []

    def get_pre_check_operators(self) -> List[InferenceOperator]:
        return []

    def get_observing_operators(self) -> List[InferenceOperator]:
        return [CheckTrainingHangOperator(self._data_manager)]

    def register_pre_check(self, pre_checks: List[Inference]):
        self._pre_checks = pre_checks

    def register_problems(self, problems: List[Inference]):
        self._training_problems = problems

    def check_training(self) -> List[Inference]:
        ic = InferenceChain(self._pre_checks, self.get_pre_check_operators())
        return ic.infer()

    def observe_training(self) -> List[Inference]:
        ic = InferenceChain(
            self._training_problems, self.get_observing_operators()
        )
        return ic.infer()

    def diagnose_failure(self, inference: Inference) -> List[Inference]:
        pass
