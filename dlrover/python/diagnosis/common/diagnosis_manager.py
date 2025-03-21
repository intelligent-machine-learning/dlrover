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
import time
from typing import Dict

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.util.function_util import TimeoutException, timeout


class DiagnosisManager:
    MIN_DIAGNOSIS_INTERVAL = 30

    def __init__(self, context):
        self._diagnosticians: Dict[str, Diagnostician] = {}
        self._periodical_diagnosis: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._context = context

    def register_diagnostician(self, name: str, diagnostician: Diagnostician):
        if diagnostician is None or len(name) == 0:
            return

        with self._lock:
            self._diagnosticians[name] = diagnostician

    def register_periodical_diagnosis(self, name: str, time_interval: int):
        with self._lock:
            if name not in self._diagnosticians:
                logger.error(f"The {name} is not registered")
                return
            if time_interval < DiagnosisManager.MIN_DIAGNOSIS_INTERVAL:
                time_interval = DiagnosisManager.MIN_DIAGNOSIS_INTERVAL

            self._periodical_diagnosis[name] = time_interval

    def start_diagnosis(self):
        with self._lock:
            logger.info(
                f"Start periodical diagnosis: {self._periodical_diagnosis}"
            )
            for name in self._periodical_diagnosis.keys():
                try:
                    thread_name = f"periodical_diagnose_{name}"
                    thread = threading.Thread(
                        target=self._start_periodical_diagnosis,
                        name=thread_name,
                        args=(name,),
                        daemon=True,
                    )
                    thread.start()
                    if thread.is_alive():
                        logger.info(f"{thread_name} initialized successfully")
                    else:
                        logger.error(f"{thread_name} is not alive")
                except Exception as e:
                    logger.error(
                        f"Failed to start the {thread_name} thread. Error: {e}"
                    )

    @timeout(secs=MIN_DIAGNOSIS_INTERVAL)
    def diagnose(self, name, **kwargs) -> DiagnosisAction:
        if name not in self._diagnosticians:
            return NoAction()
        diagnostician = self._diagnosticians[name]
        return diagnostician.diagnose(**kwargs)

    def _start_periodical_diagnosis(self, name):
        if name not in self._periodical_diagnosis:
            logger.warning(
                f"There is no periodical diagnosis registered for {name}"
            )
            return

        diagnostician = self._diagnosticians[name]
        time_interval = self._periodical_diagnosis[name]

        while True:
            time.sleep(time_interval)
            try:
                action = diagnostician.diagnose()
                if not isinstance(action, NoAction):
                    self._context.enqueue_diagnosis_action(action)
            except TimeoutException:
                logger.error(f"The diagnosis of {name} is timeout.")
            except Exception as e:
                logger.error(f"Fail to diagnose {name}: {e}")

    def observe(self, name: str, **kwargs) -> DiagnosisObservation:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.warning(f"No diagnostician is registered to observe {name}")
            return DiagnosisObservation()

        try:
            return diagnostician.observe(**kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return DiagnosisObservation()

    def resolve(
        self, name: str, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.warning(f"No diagnostician is registered to resolve {name}")
            return NoAction()

        try:
            return diagnostician.resolve(problem, **kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return NoAction()
