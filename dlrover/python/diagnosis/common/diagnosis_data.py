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

from abc import ABCMeta
from datetime import datetime
from typing import List

from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import DiagnosisDataType


class DiagnosisData(metaclass=ABCMeta):
    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
    ):
        if timestamp == 0:
            self._timestamp = int(round(datetime.now().timestamp()))
        else:
            self._timestamp = timestamp
        self._data_type = data_type
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


class WorkerDiagnosisData(DiagnosisData):
    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        node_rank: int = -1,
    ):
        """
        General metric

        Args:
            data_type (str): Type of metric. Defaults to "GENERIC".
            data_content (str): Content of the metric. Defaults to "".
            node_rank (int): Node rank. Defaults to -1.
        """

        super().__init__(timestamp, data_type, data_content)
        self._node_rank = node_rank

    @property
    def node_rank(self):
        return self._node_rank


class WorkerTrainingMetric(WorkerDiagnosisData):
    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        node_rank: int = -1,
        is_final_result=False,
        need_report=False,
    ):
        """
        General metric

        Args:
            data_type (str): Type of metric. Defaults to "GENERIC".
            data_content (str): Content of the metric. Defaults to "".
            is_final_result (bool, optional): Whether the metric is final
                result or not. Defaults to False.
            need_report (bool, optional): Whether the metric needs
                report(to Brain). Defaults to False.
            node_rank (int): Node rank. Defaults to -1.
        """

        super().__init__(timestamp, data_type, data_content, node_rank)
        self._is_final_result = is_final_result
        self._need_report = need_report

    @property
    def is_final_result(self):
        return self._is_final_result

    @property
    def need_report(self):
        return self._need_report

    def is_resolvable(self):
        if self.data_type == DiagnosisDataType.XPU_TIMER_METRIC:
            return True
        # TODO: add more resolvable metric type later
        return False


class TrainingLog(WorkerDiagnosisData):
    def __init__(self, timestamp: int = 0, logs: List[str] = None):
        if logs is None:
            data_content = ""
        else:
            data_content = "\n".join(logs)

        super().__init__(
            timestamp,
            DiagnosisDataType.TRAINING_LOG,
            data_content,
            env_utils.get_node_rank(),
        )

    @property
    def logs(self) -> List[str]:
        if not self.data_content:
            return []
        return [line for line in self.data_content.splitlines()]
