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

from typing import List

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
        hang_metric = []
        if not diagnosis_data:
            return False

        for data in diagnosis_data:
            each_metric = [
                line
                for line in data.data_content.splitlines()
                if line.startswith(HANG_METRIC_PREFIX)
            ]
            hang_metric.append(each_metric)

        if len(hang_metric) > 0:
            return True
        # TODO: implement the judgement
        return False
