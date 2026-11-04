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
from typing import Any, Dict, List, Optional

from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import DiagnosisDataType


DIAGNOSIS_DATA_SCHEMA_VERSION = 1


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
        report_id (str): Unique data report ID. Defaults to "".
        schema_version (int): Serialized report schema version. Defaults to 1.
        received_at (int): Master receive timestamp. Defaults to 0.
        sequence_id (int): Master storage sequence. Defaults to 0.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.GENERIC,
        data_content: str = "",
        node_id: int = -1,
        node_type: str = "",
        node_rank: int = -1,
        report_id: str = "",
        schema_version: int = DIAGNOSIS_DATA_SCHEMA_VERSION,
        received_at: int = 0,
        sequence_id: int = 0,
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
        self._report_id = report_id
        self._schema_version = schema_version
        self._received_at = received_at
        self._sequence_id = sequence_id

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

    @property
    def report_id(self) -> str:
        return self._report_id

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def received_at(self) -> int:
        return self._received_at

    @property
    def sequence_id(self) -> int:
        return self._sequence_id

    def bind_source(self, node_id: int, node_type: str, node_rank: int):
        """Bind the master-authoritative source of this report."""
        self._node_id = node_id
        self._node_type = node_type
        self._node_rank = node_rank

    def set_report_identity(self, report_id: str, schema_version: int):
        """Bind report identity received in the common RPC envelope."""
        self._report_id = report_id
        self._schema_version = schema_version

    def set_storage_metadata(self, received_at: int, sequence_id: int):
        """Set metadata assigned by the master after accepting a report."""
        self._received_at = received_at
        self._sequence_id = sequence_id

    def to_dict(self):
        return {k.lstrip("_"): v for k, v in self.__dict__.items()}

    def to_json(self):
        return json.dumps(self.to_dict())

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
        report_id: str = "",
        schema_version: int = DIAGNOSIS_DATA_SCHEMA_VERSION,
        received_at: int = 0,
        sequence_id: int = 0,
    ):
        super(WorkerTrainingMetric, self).__init__(
            timestamp,
            data_type,
            data_content,
            node_id,
            node_type,
            node_rank,
            report_id,
            schema_version,
            received_at,
            sequence_id,
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
        report_id: str = "",
        schema_version: int = DIAGNOSIS_DATA_SCHEMA_VERSION,
        received_at: int = 0,
        sequence_id: int = 0,
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
            report_id,
            schema_version,
            received_at,
            sequence_id,
        )

    @property
    def logs(self) -> List[str]:
        if not self.data_content:
            return []
        return [line for line in self.data_content.splitlines()]


class PerformanceDiagnosisData(DiagnosisData):
    """A worker report describing a sustained performance issue.

    The report deliberately keeps the diagnosis fields as simple JSON values.
    This makes it safe to pass through ``DiagnosisReportData`` and leaves the
    master-side diagnostician free to decide how the evidence should be
    aggregated.  ``data_content`` contains the same issue payload in JSON for
    consumers that only understand the base :class:`DiagnosisData` contract.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = DiagnosisDataType.PERFORMANCE_ISSUE,
        data_content: str = "",
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
        issue_type: str = "",
        metric_name: str = "",
        metric_value: Optional[float] = None,
        baseline_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        degradation_ratio: Optional[float] = None,
        sample_count: int = 0,
        first_timestamp: int = 0,
        last_timestamp: int = 0,
        first_iteration: Optional[int] = None,
        last_iteration: Optional[int] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        report_id: str = "",
        schema_version: int = DIAGNOSIS_DATA_SCHEMA_VERSION,
        received_at: int = 0,
        sequence_id: int = 0,
        job_uid: str = "",
        run_id: str = "",
        rdzv_epoch: int = 0,
        config_fingerprint: str = "",
    ):
        if data_content and not any(
            [
                issue_type,
                metric_name,
                metric_value is not None,
                baseline_value is not None,
                threshold_value is not None,
                degradation_ratio is not None,
                sample_count,
                first_timestamp,
                last_timestamp,
                first_iteration is not None,
                last_iteration is not None,
                evidence,
            ]
        ):
            try:
                payload = json.loads(data_content)
                issue_type = payload.get("issue_type", "")
                metric_name = payload.get("metric_name", "")
                metric_value = payload.get("metric_value")
                baseline_value = payload.get("baseline_value")
                threshold_value = payload.get("threshold_value")
                degradation_ratio = payload.get("degradation_ratio")
                sample_count = payload.get("sample_count", 0)
                first_timestamp = payload.get("first_timestamp", 0)
                last_timestamp = payload.get("last_timestamp", 0)
                first_iteration = payload.get("first_iteration")
                last_iteration = payload.get("last_iteration")
                evidence = payload.get("evidence")
            except (TypeError, ValueError):
                # Preserve an opaque payload.  The master can still retain
                # the report even when a future worker sends a new schema.
                pass

        self._issue_type = issue_type
        self._metric_name = metric_name
        self._metric_value = metric_value
        self._baseline_value = baseline_value
        self._threshold_value = threshold_value
        self._degradation_ratio = degradation_ratio
        self._sample_count = sample_count
        self._first_timestamp = first_timestamp
        self._last_timestamp = last_timestamp
        self._first_iteration = first_iteration
        self._last_iteration = last_iteration
        self._evidence = evidence or []
        self._job_uid = job_uid
        self._run_id = run_id
        self._rdzv_epoch = rdzv_epoch
        self._config_fingerprint = config_fingerprint

        if not data_content:
            data_content = json.dumps(self.issue)

        super().__init__(
            timestamp,
            data_type,
            data_content,
            node_id,
            node_type,
            node_rank,
            report_id,
            schema_version,
            received_at,
            sequence_id,
        )

    @property
    def issue_type(self):
        return self._issue_type

    @property
    def metric_name(self):
        return self._metric_name

    @property
    def metric_value(self):
        return self._metric_value

    @property
    def baseline_value(self):
        return self._baseline_value

    @property
    def threshold_value(self):
        return self._threshold_value

    @property
    def degradation_ratio(self):
        return self._degradation_ratio

    @property
    def sample_count(self):
        return self._sample_count

    @property
    def first_timestamp(self):
        return self._first_timestamp

    @property
    def last_timestamp(self):
        return self._last_timestamp

    @property
    def first_iteration(self):
        return self._first_iteration

    @property
    def last_iteration(self):
        return self._last_iteration

    @property
    def evidence(self):
        return self._evidence

    @property
    def job_uid(self):
        return self._job_uid

    @property
    def run_id(self):
        return self._run_id

    @property
    def rdzv_epoch(self):
        return self._rdzv_epoch

    @property
    def config_fingerprint(self):
        return self._config_fingerprint

    @property
    def issue(self) -> Dict[str, Any]:
        return {
            "issue_type": self._issue_type,
            "metric_name": self._metric_name,
            "metric_value": self._metric_value,
            "baseline_value": self._baseline_value,
            "threshold_value": self._threshold_value,
            "degradation_ratio": self._degradation_ratio,
            "sample_count": self._sample_count,
            "first_timestamp": self._first_timestamp,
            "last_timestamp": self._last_timestamp,
            "first_iteration": self._first_iteration,
            "last_iteration": self._last_iteration,
            "evidence": self._evidence,
        }
