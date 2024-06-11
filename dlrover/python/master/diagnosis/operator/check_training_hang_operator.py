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

from typing import List, Dict

from dlrover.python.master.diagnosis.diagnosis_data import DataManager
from dlrover.python.master.diagnosis.inferencechain.common import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)
from dlrover.python.common.log import default_logger as logger


def extract_ranks(ranks_str: str) -> List[int]:
    ranks: List[int] = []
    rank_strs = ranks_str.split("/")
    for rank_str in rank_strs:
        if "-" in rank_str:
            ss = rank_str.split("-")
            if len(ss) != 2:
                logger.error(f"invalid ranks str: {rank_str}")
                return []
            min = int(ss[0])
            max = int(ss[1])
            for i in range(min, max+1):
                ranks.append(i)
        else:
            ranks.append(int(ranks_str))
    return ranks


def is_block_func(trace: str) -> bool:
    if "synchronize@" in trace or "wait@" in trace:
        return True
    return False


def is_hanged(workers_stack: Dict[int, List[str]]) -> bool:
    blocked_ranks = []
    for rank, traces in workers_stack.items():
        blocked = True
        for i in range(0, len(traces)):
            if i == 0:
                continue
            if traces[i] != traces[i-1] or not is_block_func(traces[i]):
                blocked = False
                break
        if blocked:
            blocked_ranks.append(rank)

    if len(blocked_ranks) == len(workers_stack):
        return True
    return False


def get_latest_trace(main_traces: str) -> str:
    traces = main_traces.split(";")
    if len(traces) == 0:
        return ""
    return traces[0]


class CheckTrainingHangOperator(InferenceOperator):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self._data_manager = data_manager

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.TRAINING
            and inference.attribution == InferenceAttribute.ISORNOT
            and inference.description == InferenceDescription.HANG
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        nodes_cuda_logs = self._data_manager.get_nodes_cuda_logs()
        if not nodes_cuda_logs or len(nodes_cuda_logs) == 0:
            logger.warning("not enough data to check training hang")
            return []
        workers_latest_stack: Dict[int, List[str]] = {}

        for node_id, node_cuda_logs in nodes_cuda_logs.items():
            rank_traces: Dict[int, List[str]] = {}
            for i in range(0, 3):
                if len(node_cuda_logs) <= i:
                    continue

                for rank, main_traces in nodes_cuda_logs[i].py_main_traces.item():
                    latest_trace = get_latest_trace(main_traces)
                    if len(latest_trace) == 0:
                        logger.error(f"rank {rank} has no trace")
                        continue
                    rank_traces[rank].append(latest_trace)

        if is_hanged(workers_latest_stack):
            return [
                Inference(
                    name=InferenceName.TRAINING,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.HANG,
                )
            ]

        return []
