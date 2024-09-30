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

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import List, Optional

from dlrover.python.diagnosis.common.constants import DiagnosisDataType


class DiagnosisData(metaclass=ABCMeta):
    def __init__(self, data_type, timestamp: int = 0, data_content: str = ""):
        self._data_type = data_type
        if timestamp == 0:
            self._timestamp = int(round(datetime.now().timestamp()))
        else:
            self._timestamp = timestamp
        self._data_content = data_content

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def data_content(self) -> str:
        return self._data_content


class TrainingLog(DiagnosisData):
    def __init__(self, timestamp: int = 0, logs: List[str] = None):
        super().__init__(DiagnosisDataType.TRAINING_LOG, timestamp)
        self._logs: Optional[List[str]] = logs

    @property
    def logs(self):
        return self._logs


class AgentMetric(DiagnosisData):
    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        is_final_result=False,
        need_report=False,
        node_id: int = -1,
        node_rank: int = -1,
    ):
        """
        General metric

        Args:
            data_type (str): Type of metric. Defaults to "GENERIC".
            data_content (str): Content of the metric. Defaults to "".
            is_final_result (bool, optional): Whether the metric is final result or not. Defaults to False.
            need_report (bool, optional): Whether the metric needs report(to Brain). Defaults to False.
        """

        super().__init__(data_type, timestamp, data_content, node_id, node_rank)
        self._is_final_result = is_final_result
        self._need_report = need_report
        self._node_id = node_id
        self._node_rank = node_rank

    @property
    def is_final_result(self):
        return self._is_final_result

    @property
    def need_report(self):
        return self._need_report

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_rank(self):
        return self._node_rank

    def is_resolvable(self):
        if self.data_type == DiagnosisDataType.TRAINING_HANG_DETECTION:
            return True
        # TODO: add more resolvable metric type later
        return False
