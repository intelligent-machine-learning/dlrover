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
from typing import List, Dict

from dlrover.python.common.diagnosis import DiagnosisData, CudaLog, DiagnosisDataType
from dlrover.python.common.log import default_logger as logger


def has_expired(timestamp: float, time_period: int) -> bool:
    dt = datetime.fromtimestamp(timestamp)
    expired_dt = dt + timedelta(seconds=time_period)
    return expired_dt < datetime.now()


class NodeDataManager:
    def __init__(self, node_id: int, expire_time_period: int):
        self._node_id = node_id
        self._expire_time_period = expire_time_period
        self._diagnosis_data: Dict[str, List[DiagnosisData]] = {}

    def store_data(self, data_type: str, data: DiagnosisData):
        if data_type not in self._diagnosis_data:
            self._diagnosis_data[data_type] = []
        self._diagnosis_data[data_type].append(data)
        self._clean_diagnosis_data(data_type)
        logger.info(f"node {self._node_id} have {data_type}: {len(self._diagnosis_data[data_type])}")

    def get_data(self, data_type: str) -> List[DiagnosisData]:
        if data_type not in self._diagnosis_data:
            logger.warning(f"{data_type} is not found in the store")
            return []
        return self._diagnosis_data[data_type]

    def _clean_diagnosis_data(self, data_type: str):
        if data_type not in self._diagnosis_data:
            return

        data = self._diagnosis_data[data_type]
        n = 0
        for d in data:
            if has_expired(d.get_timestamp(), self._expire_time_period):
                n = n + 1
            else:
                break

        self._diagnosis_data[data_type] = data[n:]


class DataManager:
    def __init__(self, expire_time_period):
        self._node_data_mgrs: Dict[int, NodeDataManager] = {}
        self._expire_time_period = expire_time_period

    def store_data(self, node_id: int, data_type: str, data: DiagnosisData):
        if node_id not in self._node_data_mgrs:
            self._node_data_mgrs[node_id] = NodeDataManager(node_id, self._expire_time_period)
        self._node_data_mgrs[node_id].store_data(data_type, data)

    def get_nodes_cuda_logs(self) -> Dict[int, List[CudaLog]]:
        cuda_logs: Dict[int, List[CudaLog]] = {}
        for node_id, mgr in self._node_data_mgrs.items():
            data = mgr.get_data(DiagnosisDataType.CUDALOG)
            if data:
                cuda_logs[node_id] = data

        return cuda_logs
