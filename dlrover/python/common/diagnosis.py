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
from typing import List, Dict, Set

from dlrover.python.common.log import default_logger as logger


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
    def __init__(self, timestamp: int = 0, world_size: int = 0, traces: Dict[str, Set[int]] = None):
        super().__init__()
        if timestamp == 0:
            self._timestamp = int(round(datetime.now().timestamp()))
        else:
            self._timestamp = timestamp
        self._traces: Dict[str, Set[int]] = traces
        if not self._traces:
            self._traces = {}
        self._world_size = world_size

    def get_timestamp(self) -> int:
        return self._timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.CUDALOG

    def get_traces(self) -> Dict[str, Set[int]]:
        return self._traces

    def format_rank_trace(self) -> List[str]:
        traces: List[str] = []
        if self._world_size == 0 or len(self._traces) == 0:
            return traces
        for trace, ranks in self._traces.items():
            ranks_str = format_rank_str(self._world_size, ranks)
            rank_trace = f"{trace}#{ranks_str}"
            traces.append(rank_trace)
        return traces

    def get_world_size(self) -> int:
        return self._world_size


class TrainingLog(DiagnosisData):
    def __init__(self, timestamp: int):
        super().__init__()
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.TRAININGLOG


class ChipMetrics(DiagnosisData):
    def __init__(self, timestamp: int):
        super().__init__()
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_type(self) -> str:
        return DiagnosisDataType.CHIPMETRICES


def extract_ranks(ranks_str: str) -> List[int]:
    ranks = []
    ss = ranks_str.split("/")
    for s in ss:
        if "-" in s:
            nss = s.split("-")
            min = int(nss[0])
            max = int(nss[1])
            for i in range(min, max+1):
                ranks.append(i)
        else:
            ranks.append(int(s))

    return ranks


def format_rank_str(world_size: int, ranks: Set[int]) -> str:
    ranks = list(ranks)
    leak_ranks = list(set(range(world_size)) - set(ranks))

    def _inner_format(ranks: List[int]):
        """fold continuous ranks, [0,1,2,5,6,7]->[0-2,5-7]
        return has stack and leak stack, suppose we have 8 ranks(0-7)
        [0,1,2,5,6,7]->0-2/5-7|3-4, means rank 0-2,5-7 has this stacktrace,
        while rank 3-4 do not have this stacktrace
        """
        str_buf = []
        low = 0
        high = 0
        total = len(ranks)
        while high < total - 1:
            low_value = ranks[low]
            high_value = ranks[high]
            while high < total - 1 and high_value + 1 == ranks[high + 1]:
                high += 1
                high_value = ranks[high]
            low = high + 1
            high += 1
            if low_value != high_value:
                str_buf.append(f"{low_value}-{high_value}")
            else:
                str_buf.append(str(low_value))
        if high == total - 1:
            str_buf.append(str(ranks[high]))
        return "/".join(str_buf)

    has_stack_ranks = _inner_format(ranks)
    leak_stack_ranks = _inner_format(leak_ranks)
    return f"{has_stack_ranks}|{leak_stack_ranks}"


def node_failed(log_file: str) -> bool:
    if len(log_file) == 0:
        return False
    errors = ["error code is 507035"]
    try:
        with open(log_file, "r") as f:
            content = f.read()
            for error in errors:
                if error in content:
                    return True
    except Exception as e:
        logger.error(f"fail to read log file {log_file}: {e}")
    return False

