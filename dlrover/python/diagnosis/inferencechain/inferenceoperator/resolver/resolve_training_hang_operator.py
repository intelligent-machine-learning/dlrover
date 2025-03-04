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
from typing import List

from dlrover.python.common.constants import EventReportConstants
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
    is_same_inference,
)

_dlrover_ctx = Context.singleton_instance()


class ResolveTrainingHangOperator(InferenceOperator):
    """
    CheckTrainingHangOperator is the operator to check
    if training is hanged
    """

    def __init__(self, data_manager):
        super().__init__(data_manager)

    def is_compatible(self, inference: Inference) -> bool:
        if inference.name == InferenceName.TRAINING:
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        hang_infer = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.HANG,
        )
        if any(is_same_inference(infer, hang_infer) for infer in inferences):
            if _dlrover_ctx.hang_detection == 1:
                # trigger event action
                return [
                    Inference(
                        name=InferenceName.ACTION,
                        attribution=InferenceAttribute.IS,
                        description=InferenceDescription.EVENT,
                        configs={
                            "event_type": EventReportConstants.TYPE_WARN,
                            "event_instance": EventReportConstants.JOB_INSTANCE,  # noqa: E501
                            "event_action": EventReportConstants.ACTION_HANG_WARN,  # noqa: E501
                            "event_msg": "",
                            "event_labels": json.dumps({}),
                        },
                    )
                ]
            elif _dlrover_ctx.hang_detection == 2:
                # trigger relaunch action
                pass

        return [
            Inference(
                name=InferenceName.ACTION,
                attribution=InferenceAttribute.IS,
                description=InferenceDescription.NONE,
            )
        ]
