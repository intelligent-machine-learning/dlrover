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

import sys
import threading
from collections import deque
from itertools import islice
from typing import Dict, List

from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.util.time_util import has_expired


class DiagnosisDataManager:
    """
    DiagnosisDataManager is to manage the diagnosis data collected from worker
    """

    def __init__(self, expire_time_period=600):
        """
        Args:
            expire_time_period: data expire time period in seconds
        """
        self._diagnosis_data: Dict[str, deque[DiagnosisData]] = {}
        self._expire_time_period = expire_time_period
        self._job_context = get_job_context()
        self._lock = threading.Lock()

    @property
    def data(self):
        return self._diagnosis_data

    @property
    def expire_time_period(self):
        return self._expire_time_period

    def store_data(self, data: DiagnosisData):
        data_type = data.data_type
        with self._lock:
            if data_type not in self.data:
                self.data[data_type] = deque(maxlen=100000)
            self.data[data_type].append(data)
            self._clean_diagnosis_data(data_type)

    def get_data(self, data_type: str) -> List[DiagnosisData]:
        with self._lock:
            if data_type not in self.data:
                return []
            return list(self.data[data_type])

    def get_data_size(self):
        return sys.getsizeof(self.data)

    def _clean_diagnosis_data(self, data_type: str):
        if data_type not in self.data:
            return

        each_data = self.data[data_type]
        n = 0
        for d in each_data:
            if has_expired(d.timestamp, self.expire_time_period):
                n = n + 1
            else:
                break
        if n > 0:
            self.data[data_type] = deque(islice(each_data, n, len(each_data)))
