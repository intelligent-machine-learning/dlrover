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
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisErrorConstant,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)


class ResolveGPUErrorsOperator(InferenceOperator):
    """
    ResolveGPUErrorsOperator is to diagnose GPU errors like:
    1. GPU lost. This error will be reported to the master and
    the master will expose this error at this moment.
    """

    def __init__(self):
        super().__init__(None)

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.GPU
            and inference.attribution == InferenceAttribute.IS
            and inference.description == InferenceDescription.ERROR
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        if (
            len(inferences) != 1
            or not inferences[0].configs
            or InferenceConfigKey.LOGS not in inferences[0].configs
            or InferenceConfigKey.ERRORS not in inferences[0].configs
        ):
            if len(inferences) > 0 and inferences[0].configs:
                logger.error(
                    f"invalid configurations: {inferences[0].configs}"
                )
            else:
                logger.error("no configurations found")
            return []
        inf = inferences[0]
        if (
            inf.configs[InferenceConfigKey.ERRORS]
            == DiagnosisErrorConstant.GPU_LOST
        ):
            return [
                Inference(
                    name=InferenceName.ACTION,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.EVENT,
                    configs={
                        InferenceConfigKey.EVENT_TYPE: EventReportConstants.TYPE_WARN,  # noqa: E501
                        InferenceConfigKey.EVENT_INSTANCE: f"{DiagnosisConstant.LOCAL_INSTANCE}",  # noqa: E501
                        InferenceConfigKey.EVENT_ACTION: inf.configs[
                            InferenceConfigKey.ERRORS
                        ],
                        InferenceConfigKey.EVENT_MSG: inf.configs[
                            InferenceConfigKey.LOGS
                        ],
                        InferenceConfigKey.EVENT_LABELS: json.dumps({}),
                        InferenceConfigKey.EXPIRED_TIME_PERIOD: "120",
                    },
                )
            ]

        return []
