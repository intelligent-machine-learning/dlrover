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

from dlrover.python.common.diagnosis import CudaLog
from dlrover.python.elastic_agent.datacollector.data_collector import DataCollector
from typing import List, Dict
from pathlib import Path
from dlrover.python.common.log import default_logger as logger
from collections import defaultdict
from dlrover.proto import elastic_training_pb2


class CudaLogCollector(DataCollector):
    def __init__(self, log_path: str):
        super().__init__()
        self._world_size = 0
        self._log_path = log_path
        self._stack_count = defaultdict(int)
        self._stack_rank_count = defaultdict(set)
        self._ranks = None
        self._py_main_thread_traces: Dict[int, str] = {}

    def collect_data(self) -> CudaLog:
        logs = Path(self._log_path)
        log_files = sorted(logs.glob("*stacktrace"))
        if not log_files:
            logger.info(f"there is no cuda log files at {self._log_path}")
            return CudaLog(0, {})
        self._world_size = int(log_files[0].name[6:11])
        self._ranks = set(range(self._world_size))
        logger.info(f"world_size: {self._world_size}, ranks: {self._ranks}")

        self._py_main_thread_traces = {}
        self._parse("py", log_files)
        return CudaLog(0, self._py_main_thread_traces)

    def to_collect_data(self) -> bool:
        return True

    def _parse(self, mode, files) -> List[str]:
        self._stack_count = defaultdict(int)
        self._stack_rank_count = defaultdict(set)
        for file in files:
            self._parse_one(file.name, mode)
        rank_traces: List[str] = []

        # for trace, count in self._stack_count.items():
        #     rank_str = self._format_rank_str(self._stack_rank_count[trace])
        #     if trace:
        #         rank_trace = f"{trace}@{rank_str}\n"
        #         rank_traces.append(rank_trace)
        return rank_traces

    def _parse_one(self, file_name, mode):
        st = elastic_training_pb2.Stacktrace()
        # 00003-00008.stacktrace
        rank = int(file_name[:5])
        file_path = self._log_path + "/" + file_name
        logger.info(f"parse rank {rank} log: {file_path}")
        with open(file_path, "rb") as f:
            st.ParseFromString(f.read())
        if mode == "cpp":
            self._frame_hash(st.stacktrace, rank)
        else:
            self._frame_hash(st.py_stacktrace, rank)

    def _frame_hash(self, traces, rank):
        for trace in traces:
            buf = []
            for index, frame in enumerate(trace.frames[::-1]):
                func_file_name = f"{frame.func_name}@{frame.file_name}"
                buf.append(func_file_name)
            stackchain = f"{';'.join(buf)}"
            self._stack_rank_count[stackchain].add(rank)
            self._stack_count[stackchain] = 1
            if trace.thread_name == "MainThread":
                self._py_main_thread_traces[rank] = stackchain
                logger.info(f"rank {rank} main trace: {stackchain}")

    def _format_rank_str(self, ranks):
        ranks = list(ranks)
        leak_ranks = list(self._ranks - set(ranks))

        def _inner_format(ranks):
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
