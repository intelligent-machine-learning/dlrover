import threading
from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from typing import Dict
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
)
from dlrover.python.common.log import default_logger as logger


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
            return diagnostician.observe(**kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return NoAction()

    def resolve(self, name: str, problem: DiagnosisAction, **kwargs) -> DiagnosisAction:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.error(f"No diagnostician is registered to resolve {name}")
            return NoAction()

        try:
            return diagnostician.resolve(problem, **kwargs)
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return NoAction()