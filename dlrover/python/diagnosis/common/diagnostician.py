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

from abc import ABCMeta

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)


class DiagnosisObservation(metaclass=ABCMeta):
    """
    DiagnosisObservation is to describe the problem observed
    by Diagnostician.observe
    """

    def __init__(self, observation: str = ""):
        # The simple description info for the problem.
        self._observation: str = observation

    @property
    def observation(self):
        return self._observation


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

    def observe(self, **kwargs) -> DiagnosisObservation:
        # observe if particular problem happened
        return DiagnosisObservation("unknown")

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        # explore the solution to resolve the problem
        return EventAction()

    def diagnose(self, **kwargs) -> DiagnosisAction:
        # define the diagnosis procedure
        try:
            ob = self.observe(**kwargs)
            return self.resolve(ob, **kwargs)
        except Exception as e:
            logger.error(f"Fail to diagnose the problem: {e}")
            return NoAction()
