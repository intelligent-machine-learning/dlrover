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

from datetime import datetime, timedelta
from typing import List

from dlrover.python.common.diagnosis import DiagnosisData
from dlrover.python.common.log import default_logger as logger


def has_expired(timestamp: float, time_period: int) -> bool:
    dt = datetime.fromtimestamp(timestamp)
    expired_dt = dt + timedelta(seconds=time_period)
    return expired_dt < datetime.now()


class DataManager:
    def __init__(self, expire_time_period):
        self.diagnosis_data: dict[str, [DiagnosisData]] = {}
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
