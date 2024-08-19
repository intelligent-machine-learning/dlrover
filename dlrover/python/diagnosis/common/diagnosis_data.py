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


class DiagnosisDataType:
    CUDALOG = "cuda_log"
    TRAININGLOG = "training_log"
    CHIPMETRICES = "chip_metrics"


class DiagnosisData(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_timestamp(self) -> float:
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass


class CudaLog(DiagnosisData):
    def __init__(self, timestamp: int):
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.CUDALOG


class TrainingLog(DiagnosisData):
    def __init__(self, timestamp: int = 0, logs: List[str] = None):
        super().__init__()
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp
        self.logs: Optional[List[str]] = logs

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.TRAININGLOG


class ChipMetrics(DiagnosisData):
    def __init__(self, timestamp: int):
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.CHIPMETRICES
