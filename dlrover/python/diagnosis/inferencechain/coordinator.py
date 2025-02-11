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

from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    is_same_inference,
)

_dlrover_ctx = Context.singleton_instance()


def coordinate_solutions(
    solutions: List[Inference],
) -> DiagnosisAction:
    """
    Transform solutions (of Inference) to executable diagnosis action

    Args:
        solutions: solutions of Inference
    Return:
        diagnosis action
    """

    if len(solutions) == 0:
        return NoAction()

    logger.info(f"coordinate solutions: {solutions}")
    event_solution = Inference(
        name=InferenceName.ACTION,
        attribution=InferenceAttribute.IS,
        description=InferenceDescription.EVENT,
    )
    for solution in solutions:
        # deal with event
        if is_same_inference(solution, event_solution):
            event_payload = solution.configs

            expired_time_period = (
                DiagnosisConstant.ACTION_EXPIRED_TIME_PERIOD_DEFAULT
            )
            if InferenceConfigKey.EXPIRED_TIME_PERIOD in event_payload:
                expired_time_period = int(
                    event_payload[InferenceConfigKey.EXPIRED_TIME_PERIOD]
                )
            executable_time_period = 0
            if InferenceConfigKey.EXECUTABLE_TIME_PERIOD in event_payload:
                executable_time_period = int(
                    event_payload[InferenceConfigKey.EXECUTABLE_TIME_PERIOD]
                )

            return EventAction(
                event_type=event_payload[InferenceConfigKey.EVENT_TYPE],
                event_instance=event_payload[
                    InferenceConfigKey.EVENT_INSTANCE
                ],
                event_action=event_payload[InferenceConfigKey.EVENT_ACTION],
                event_msg=event_payload[InferenceConfigKey.EVENT_MSG],
                event_labels=json.loads(
                    event_payload[InferenceConfigKey.EVENT_LABELS]
                ),
                expired_time_period=expired_time_period,
                executable_time_period=executable_time_period,
            )

    return NoAction()

