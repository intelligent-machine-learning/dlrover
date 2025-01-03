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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
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


class CheckResourceCollectionOperator(InferenceOperator):
    """
    CheckResourceCollectionOperator is to analyze the errors
    during resource collection
    """

    def __init__(self):
        super().__init__(None)

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.RESOURCE
            and inference.attribution == InferenceAttribute.COLLECT
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
        ):
            if len(inferences) > 0 and inferences[0].configs:
                logger.error(
                    f"invalid configurations: {inferences[0].configs}"
                )
            else:
                logger.error("no configurations found")
            return []
        logs = inferences[0].configs[InferenceConfigKey.LOGS]
        if DiagnosisErrorConstant.GPU_LOST in logs:
            return [
                Inference(
                    name=InferenceName.GPU,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.ERROR,
                    configs={
                        InferenceConfigKey.LOGS: logs,
                        InferenceConfigKey.ERRORS: DiagnosisErrorConstant.GPU_LOST,  # noqa: E501
                    },
                )
            ]
        return []
