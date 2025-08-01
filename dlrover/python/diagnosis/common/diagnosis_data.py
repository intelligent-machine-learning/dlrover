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

import json
from abc import ABCMeta
from datetime import datetime
from typing import List, Optional

from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import DiagnosisDataType


class DiagnosisData(metaclass=ABCMeta):
    """
    Basic definition of diagnosis data.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        data_type (str): Type of metric. Defaults to "GENERIC".
        data_content (str): Content of the metric. Defaults to "".
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        node_id: int = -1,
        node_type: str = "",
        node_rank: int = -1,
    ):
        if timestamp == 0:
            self._timestamp = int(round(datetime.now().timestamp()))
        else:
            self._timestamp = timestamp
        self._data_type = data_type
        self._data_content = data_content
        self._node_id = node_id
        self._node_type = node_type
        self._node_rank = node_rank

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def data_content(self) -> str:
        return self._data_content

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type(self):
        return self._node_type

    @property
    def node_rank(self):
        return self._node_rank

    def to_json(self):
        data = {k.lstrip("_"): v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))

    def is_from_worker(self):
        return self._node_id != -1


class WorkerTrainingMetric(DiagnosisData):
    """
    Diagnosis data for worker training metric.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        data_type (str): Type of metric. Defaults to "GENERIC".
        data_content (str): Content of the metric. Defaults to "".
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
        is_final_result (bool, optional): Whether the metric is final result.
            Defaults to False.
        need_report (bool, optional): Whether the metric needs report.
            Defaults to False.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
        is_final_result=False,
        need_report=False,
    ):
        super(WorkerTrainingMetric, self).__init__(
            timestamp, data_type, data_content, node_id, node_type, node_rank
        )
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


class TrainingLog(DiagnosisData):
    """
    Worker's training log.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        logs (list): Log content in list format.
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        logs: Optional[List[str]] = None,
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
    ):
        if logs is None:
            data_content = ""
        else:
            data_content = "\n".join(logs)

        super().__init__(
            timestamp,
            DiagnosisDataType.TRAINING_LOG,
            data_content,
            node_id,
            node_type,
            node_rank,
        )

    @property
    def logs(self) -> List[str]:
        if not self.data_content:
            return []
        return [line for line in self.data_content.splitlines()]
