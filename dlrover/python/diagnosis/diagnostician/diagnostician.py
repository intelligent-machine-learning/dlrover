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

import threading
from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)


class Diagnostician(metaclass=ABCMeta):
    """
    Diagnostician is to observe problems and resolve those problems
    """

    def __init__(self):
        pass

    @abstractmethod
    def observe(self, **kwargs) -> DiagnosisAction:
        # observe if particular problem happened
        return NoAction()

    @abstractmethod
    def resolve(self, problem: DiagnosisAction, **kwargs) -> DiagnosisAction:
        # explore the solution to resolve the
        return NoAction()


class DiagnosticianManager:
    def __init__(self):
        self._diagnosticians: Dict[str, Diagnostician] = {}
        self._lock = threading.Lock()

    def register_diagnostician(self, name: str, diagnostician: Diagnostician):
        if diagnostician is None or len(name) == 0:
            return

        with self._lock:
            self._diagnosticians[name] = diagnostician

    def observe(self, name: str, **kwargs) -> DiagnosisAction:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.error(f"No diagnostician is registered to observe {name}")
            return NoAction()

        try:
            print(f"observe {name}\n")
            return diagnostician.observe(**kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return NoAction()

    def resolve(
        self, name: str, problem: DiagnosisAction, **kwargs
    ) -> DiagnosisAction:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.error(f"No diagnostician is registered to resolve {name}")
            return NoAction()

        try:
            print(f"resolve {name}\n")
            return diagnostician.resolve(problem, **kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return NoAction()
