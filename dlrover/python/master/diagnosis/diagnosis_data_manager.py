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

import hashlib
import json
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from dlrover.python.diagnosis.common.constants import DiagnosisDataType
from dlrover.python.diagnosis.common.diagnosis_data import (
    DIAGNOSIS_DATA_SCHEMA_VERSION,
    DiagnosisData,
    PerformanceDiagnosisData,
)
from dlrover.python.master.node.job_context import get_job_context


STORE_ACCEPTED = "accepted"
STORE_DUPLICATE = "duplicate"
STORE_CONFLICT = "conflict"
STORE_REJECTED = "rejected"

MAX_DIAGNOSIS_REPORT_SIZE = 256 * 1024


class DiagnosisDataManager:
    """
    Manage bounded, deduplicated diagnosis data collected from workers.
    """

    def __init__(
        self,
        expire_time_period=600,
        max_records_per_type=100000,
        max_total_bytes=16 * 1024 * 1024,
        max_report_size=MAX_DIAGNOSIS_REPORT_SIZE,
    ):
        """
        Args:
            expire_time_period: data expire time period in seconds
            max_records_per_type: maximum retained reports for one data type
            max_total_bytes: maximum retained payload bytes for one data type
            max_report_size: maximum serialized size of one report
        """
        self._diagnosis_data: Dict[str, Deque[DiagnosisData]] = {}
        self._report_index: Dict[str, DiagnosisData] = {}
        self._data_bytes: Dict[str, int] = {}
        self._latest_sequence_ids: Dict[str, int] = {}
        self._next_sequence_id = 0
        self._expire_time_period = expire_time_period
        self._max_records_per_type = max(1, max_records_per_type)
        self._max_report_size = max(1, max_report_size)
        self._max_total_bytes = max(self._max_report_size, max_total_bytes)
        self._job_context = get_job_context()
        self._lock = threading.Lock()

    @property
    def data(self):
        return self._diagnosis_data

    @property
    def expire_time_period(self):
        return self._expire_time_period

    @property
    def max_report_size(self):
        return self._max_report_size

    def store_data(
        self,
        data: DiagnosisData,
        node_id: Optional[int] = None,
        node_type: Optional[str] = None,
        node_rank: Optional[int] = None,
    ):
        if not isinstance(data, DiagnosisData):
            return STORE_REJECTED

        if data.schema_version != DIAGNOSIS_DATA_SCHEMA_VERSION:
            return STORE_REJECTED

        if (
            data.data_type == DiagnosisDataType.PERFORMANCE_ISSUE
            and not isinstance(data, PerformanceDiagnosisData)
        ):
            return STORE_REJECTED
        if (
            isinstance(data, PerformanceDiagnosisData)
            and data.data_type != DiagnosisDataType.PERFORMANCE_ISSUE
        ):
            return STORE_REJECTED

        if data.data_type == DiagnosisDataType.PERFORMANCE_ISSUE and (
            node_id is None or node_id < 0
        ):
            return STORE_REJECTED

        if node_id is not None:
            data.bind_source(
                node_id,
                node_type or "",
                node_rank if node_rank is not None else -1,
            )

        report_size = self._report_size(data)
        if report_size > self._max_report_size:
            return STORE_REJECTED

        if (
            data.data_type == DiagnosisDataType.PERFORMANCE_ISSUE
            and not data.report_id
        ):
            return STORE_REJECTED

        payload_hash = self._payload_hash(data)
        data_type = data.data_type

        with self._lock:
            reports = self._reports_locked(data_type)
            self._clean_diagnosis_data_locked(data_type)

            if data.report_id:
                existing = self._report_index.get(data.report_id)
                if existing is not None:
                    if self._payload_hash(existing) == payload_hash:
                        return STORE_DUPLICATE
                    return STORE_CONFLICT

            received_at = int(time.time())
            self._next_sequence_id += 1
            data.set_storage_metadata(received_at, self._next_sequence_id)
            reports.append(data)
            self._data_bytes[data_type] += report_size
            self._latest_sequence_ids[data_type] = data.sequence_id
            if data.report_id:
                self._report_index[data.report_id] = data

            self._enforce_limits_locked(data_type)
            return STORE_ACCEPTED

    def get_data(
        self,
        data_type: str,
        after_sequence_id: int = 0,
        limit: Optional[int] = None,
    ) -> List[DiagnosisData]:
        with self._lock:
            if data_type not in self.data:
                return []
            self._clean_diagnosis_data_locked(data_type)
            reports = [
                data
                for data in self.data[data_type]
                if data.sequence_id > after_sequence_id
            ]
            if limit is not None:
                reports = reports[: max(0, limit)]
            return reports

    def get_latest_sequence_id(self, data_type: str) -> int:
        with self._lock:
            if data_type not in self.data:
                return self._latest_sequence_ids.get(data_type, 0)
            self._clean_diagnosis_data_locked(data_type)
            return self._latest_sequence_ids.get(data_type, 0)

    def get_data_size(self):
        with self._lock:
            return sum(self._data_bytes.values())

    def _reports_locked(self, data_type: str) -> Deque[DiagnosisData]:
        if data_type not in self.data:
            self.data[data_type] = deque()
            self._data_bytes[data_type] = 0
            self._latest_sequence_ids.setdefault(data_type, 0)
        return self.data[data_type]

    def _clean_diagnosis_data_locked(self, data_type: str):
        if data_type not in self.data:
            return

        reports = self.data[data_type]
        expired_before = time.time() - self.expire_time_period
        while reports:
            received_at = reports[0].received_at or reports[0].timestamp
            if received_at > expired_before:
                break
            self._remove_oldest_locked(data_type)

    def _enforce_limits_locked(self, data_type: str):
        reports = self.data[data_type]
        while len(reports) > self._max_records_per_type:
            self._remove_oldest_locked(data_type)
        while self._data_bytes[data_type] > self._max_total_bytes and reports:
            self._remove_oldest_locked(data_type)

    def _remove_oldest_locked(self, data_type: str):
        old_data = self.data[data_type].popleft()
        self._data_bytes[data_type] -= self._report_size(old_data)
        if (
            old_data.report_id
            and self._report_index.get(old_data.report_id) is old_data
        ):
            del self._report_index[old_data.report_id]

    @staticmethod
    def _payload_without_storage_metadata(data: DiagnosisData):
        payload = data.to_dict()
        payload.pop("received_at", None)
        payload.pop("sequence_id", None)
        return payload

    @classmethod
    def _report_size(cls, data: DiagnosisData) -> int:
        payload = cls._payload_without_storage_metadata(data)
        return len(
            json.dumps(
                payload,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )

    @classmethod
    def _payload_hash(cls, data: DiagnosisData) -> str:
        payload = cls._payload_without_storage_metadata(data)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
