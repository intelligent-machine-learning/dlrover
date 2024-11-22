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

import re
import sys
from typing import Dict, List, Tuple

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisDataType
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)

HANG_METRIC_PREFIX = "XPU_TIMER_COMMON_HANG"


class CheckTrainingHangOperator(InferenceOperator):
    """
    CheckTrainingHangOperator is the operator to check
    if training is hanged
    """

    def __init__(self, data_manager):
        super().__init__(data_manager)

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
        if not self.data_manager:
            logger.info(
                "Skip training-hang inference for there is "
                "no diagnosis data."
            )
            return [
                Inference(
                    name=InferenceName.TRAINING,
                    attribution=InferenceAttribute.NOT,
                    description=InferenceDescription.HANG,
                )
            ]

        diagnosis_data = self._data_manager.get_data(
            DiagnosisDataType.XPU_TIMER_METRIC
        )

        if diagnosis_data and self.is_hang(diagnosis_data):
            logger.warning("Training might hanged.")
            return [
                Inference(
                    name=InferenceName.TRAINING,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.HANG,
                )
            ]

        return [
            Inference(
                name=InferenceName.TRAINING,
                attribution=InferenceAttribute.NOT,
                description=InferenceDescription.HANG,
            )
        ]

    def is_hang(self, diagnosis_data: List[DiagnosisData]):
        logger.debug(
            "Hang detection start using diagnosis data, "
            f"data number: {len(diagnosis_data)}, "
            f"data size: {sys.getsizeof(diagnosis_data)}."
        )
        worker_hang_metric: Dict[int, List[Tuple[int, bool]]] = {}
        if not diagnosis_data:
            logger.debug("Skip for no worker hang metric.")
            return False

        # the format of the hang metric can refer these files:
        # dlrover/python/tests/data/xpu_timer/hang
        for data in diagnosis_data:
            # filter hang metric
            each_worker_metric = [
                line
                for line in data.data_content.splitlines()
                if line.startswith(HANG_METRIC_PREFIX)
            ]

            # if all local rank is hanged, tag worker hang
            rank_hang_size = 0
            is_worker_hang = False
            for each_rank_metric in each_worker_metric:
                match = re.search(r"(\d+)(?!.*\d)", each_rank_metric)
                if match and match.group(0) == "1":
                    rank_hang_size += 1
            if rank_hang_size == len(each_worker_metric):
                is_worker_hang = True

            if data.node_rank not in worker_hang_metric:
                worker_hang_metric[data.node_rank] = []
            worker_hang_metric[data.node_rank].append(
                (data.timestamp, is_worker_hang)
            )

        # hang detection rules:
        # 1. 100% worker got hang metric
        # 2. last for 5+ minutes
        hang_id, hang_last = self._get_hang_overlaps(worker_hang_metric)
        hang_last_threshold = self._get_hang_time_last_threshold()
        if hang_id != -1 and hang_last > hang_last_threshold:
            logger.info(
                f"Got hang worker: {hang_id}, time last: {hang_last}, "
                f"threshold: {hang_last_threshold}"
            )
            return True
        return False

    def _get_hang_time_last_threshold(self):
        # set 5 minutes for now(second)
        return 5 * 60

    def _get_hang_overlaps(
        self, worker_hang_metric: Dict[int, List[Tuple[int, bool]]]
    ) -> Tuple[int, int]:
        """
        Require all workers hang from latest and find the hang overlaps.

        Args:
            worker_hang_metric (Dict[int, List[Tuple[int, bool]]]): Input
                metric in format: node_id: [(timestamp, is_hang), ...]

        Returns:
            The hang overlaps' id and time last in tuple format.
        """

        worker_hang_length_min = 0
        worker_hang_id = -1

        # find the intersection from latest
        for worker_id, tuple_list in worker_hang_metric.items():
            # sorted by timestamp
            tuple_list.sort(key=lambda x: x[0])
            worker_hang_length = 0

            # ort in descending order
            reversed_tuple_list = reversed(tuple_list)
            for tuple_item in reversed_tuple_list:
                if tuple_item[1]:
                    worker_hang_length += 1
                else:
                    break

            if worker_hang_length > 0:
                if worker_hang_length_min == 0:
                    worker_hang_length_min = worker_hang_length
                    worker_hang_id = worker_id
                elif worker_hang_length < worker_hang_length_min:
                    worker_hang_length_min = worker_hang_length
                    worker_hang_id = worker_id
            else:
                # there is normal worker
                return -1, -1

        # get the intersection's time last
        if worker_hang_id != -1 and worker_hang_length_min != 0:
            hang_worker_metric = worker_hang_metric[worker_hang_id]
            time_last = (
                hang_worker_metric[len(hang_worker_metric) - 1][0]
                - hang_worker_metric[
                    len(hang_worker_metric) - worker_hang_length_min
                ][0]
            )
            return worker_hang_id, time_last

        return -1, -1
