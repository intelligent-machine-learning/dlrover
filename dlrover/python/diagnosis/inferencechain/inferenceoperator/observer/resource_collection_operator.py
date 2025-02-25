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
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor


class ResourceCollectionOperator(InferenceOperator):
    """
    ResourceCollectionOperator is the operator to collect
    worker resources.
    """

    def __init__(self):
        super().__init__(None)
        self._monitor = ResourceMonitor().singleton_instance()

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.WORKER
            and inference.attribution == InferenceAttribute.COLLECT
            and inference.description == InferenceDescription.RESOURCE
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        error_logs = ""
        try:
            self._monitor.report_resource()
        except Exception as e:
            error_logs = f"{e}"

        if DiagnosisErrorConstant.GPU_LOST in error_logs:
            return [
                Inference(
                    name=InferenceName.GPU,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.ERROR,
                    configs={
                        InferenceConfigKey.LOGS: error_logs,
                        InferenceConfigKey.ERRORS: DiagnosisErrorConstant.GPU_LOST,  # noqa: E501
                    },
                ),
            ]

        return []
