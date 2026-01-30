# Copyright 2025 The DLRover Authors. All rights reserved.
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
from typing import Optional

from dlrover.python.common.constants import DictKey, EventReportConstants
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisErrorConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)


class ResourceCollectionFailureDiagnostician(Diagnostician):
    """
    ResourceCollectionFailureDiagnostician is to collect and report resource
    to the master and handle the errors during collection
    """

    def __init__(self):
        super().__init__()

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        error_log_arg = kwargs.get("error_log")
        error_log = str(error_log_arg)

        if DiagnosisErrorConstant.GPU_LOST in error_log:
            ob = DiagnosisObservation(
                DiagnosisErrorConstant.GPU_LOST, {DictKey.LOGS: error_log}
            )
            ob.extra_infos[DictKey.LOGS] = error_log
            return ob

        return None

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        if problem.observation == DiagnosisErrorConstant.GPU_LOST:
            return EventAction(
                event_type=EventReportConstants.TYPE_WARN,
                event_instance=f"{DiagnosisConstant.LOCAL_INSTANCE}",
                event_action=problem.observation,
                event_msg=problem.extra_infos.get(DictKey.LOGS, ""),
                event_labels={},
                expired_time_period=120,
            )
        else:
            return NoAction()
