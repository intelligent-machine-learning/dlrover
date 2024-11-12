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
from collections import deque
from typing import Deque, Dict, List

from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.util.time_util import has_expired


class DiagnosisDataManager:
    """
    DiagnosisDataManager is to manage the diagnosis data collected from worker
    """

    def __init__(self, expire_time_period):
        """
        Args:
            expire_time_period: data expire time period in seconds
        """
        self.diagnosis_data: Dict[str, deque[DiagnosisData]] = {}
        self.expire_time_period = expire_time_period
        self._lock = threading.Lock()

    def store_data(self, data: DiagnosisData):
        data_type = data.data_type
        with self._lock:
            if data_type not in self.diagnosis_data:
                self.diagnosis_data[data_type] = deque(maxlen=100)
            q = self.diagnosis_data[data_type]
            q.append(data)

    def get_data(self, data_type: str) -> List[DiagnosisData]:
        with self._lock:
            if data_type not in self.diagnosis_data:
                return []
            data_by_type = self.diagnosis_data[data_type]
            result = []
            while len(data_by_type) != 0:
                data = data_by_type.popleft()
                if not has_expired(data.timestamp, self.expire_time_period):
                    result.append(data)
            return result
