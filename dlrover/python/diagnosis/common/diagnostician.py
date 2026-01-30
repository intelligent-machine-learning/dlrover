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

from typing import Dict, Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)
from dlrover.python.util.function_util import (
    TimeoutException,
    threading_timeout,
)


class DiagnosisObservation:
    """
    DiagnosisObservation is to describe the problem observed
    by Diagnostician.observe
    """

    def __init__(
        self, observation: str = "", extra_infos: Dict[str, str] = {}
    ):
        # The simple description info for the problem.
        self._observation: str = observation
        self._extra_infos: Dict[str, str] = extra_infos

    @property
    def observation(self):
        return self._observation

    @property
    def extra_infos(self):
        return self._extra_infos


class Diagnostician:
    """
    Diagnostician is to observe problems and resolve those problems.

    It includes three APIs here:
    1. observe: observe if one particular problem happened or not
    2. resolve: generates the DiagnosisAction to handle the problem
    observed.
    3. diagnose: define the procedure of the whole diagnosis for
    a particular problem.
    """

    def __init__(self):
        pass

    @threading_timeout(secs=DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL)
    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        # observe if particular problem happened
        return DiagnosisObservation("unknown")

    @threading_timeout(secs=DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL)
    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        # explore the solution to resolve the problem
        return EventAction()

    def diagnose(self, **kwargs) -> DiagnosisAction:
        # define the diagnosis procedure
        try:
            ob = self.observe(**kwargs)
            if ob:
                return self.resolve(ob, **kwargs)
            return NoAction()
        except TimeoutException:
            logger.error(
                f"The diagnosis of {self.__class__.__name__} is timeout."
            )
            return NoAction()
        except Exception as e:
            logger.error(f"Fail to diagnose the problem: {e}")
            return NoAction()
