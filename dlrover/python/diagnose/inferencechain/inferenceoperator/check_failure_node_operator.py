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
from dlrover.python.diagnose.common.constants import InferenceConfigKey
from dlrover.python.diagnose.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)
from dlrover.python.diagnose.datacollector.training_log_collector import (
    TrainingLogCollector,
)


class CheckFailureNodeOperator(InferenceOperator):
    def __init__(self):
        super().__init__()

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.NODE
            and inference.attribution == InferenceAttribute.ISORNOT
            and inference.description == InferenceDescription.FAILURE
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        if (
            len(inferences) == 0
            or not inferences[0].configs
            or InferenceConfigKey.LOG_FILE not in inferences[0].configs
            or InferenceConfigKey.ERRORS not in inferences[0].configs
        ):
            return [
                Inference(
                    name=InferenceName.NODE,
                    attribution=InferenceAttribute.NOT,
                    description=InferenceDescription.FAILURE,
                )
            ]
        log_file = inferences[0].configs[InferenceConfigKey.LOG_FILE]
        errors = inferences[0].configs[InferenceConfigKey.ERRORS]
        error_codes = errors.split("#")

        collector = TrainingLogCollector(log_file, 5000)
        training_log = collector.collect_data()
        logs = training_log.logs
        if not logs or len(logs) == 0:
            logger.error(f"fail to collect training log from {log_file}")
            return [
                Inference(
                    name=InferenceName.NODE,
                    attribution=InferenceAttribute.NOT,
                    description=InferenceDescription.FAILURE,
                )
            ]

        is_failure_node = False
        for log in logs:
            if is_failure_node:
                break
            for error in error_codes:
                if error in log:
                    is_failure_node = True
                    break
        if is_failure_node:
            return [
                Inference(
                    name=InferenceName.NODE,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.FAILURE,
                )
            ]
        return [
            Inference(
                name=InferenceName.NODE,
                attribution=InferenceAttribute.NOT,
                description=InferenceDescription.FAILURE,
            )
        ]
