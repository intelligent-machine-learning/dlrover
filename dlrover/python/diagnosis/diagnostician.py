from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.trouble_shooter import TroubleShooter
from typing import Dict, List
import threading
from dlrover.python.common.log import default_logger as logger

class Diagnostician:
    """
    Diagnostician is to observe training problems and explore solutions to
    those problems during training.
    """

    def __init__(self):
        self._observers: Dict[str, Observer] = {}
        self._shooters: List[TroubleShooter] = []
        self._lock = threading.Lock()

    def observe(self, problem: str) -> DiagnosisAction:
        with self._lock:
            observer = self._observers.get(problem, None)
        if observer is None:
            return NoAction()
        try:
            return observer.observe()
        except Exception as e:
            logger.error(f"fail to execute observer {observer.__class__}: {e}")
            return NoAction()

    def register_observer(self, name: str, observer: Observer):
        with self._lock:
            if len(name) == 0:
                return
            self._observers[name] = observer

    def register_trouble_shooter(self, shooter: TroubleShooter):
        with self._lock:
            if shooter is None:
                return
            self._shooters.append(shooter)

    def trouble_shoot(self, action: DiagnosisAction) -> DiagnosisAction:
        for shooter in self._shooters:
            if shooter.is_compatible(action):
                try:
                    return shooter.trouble_shoot(action)
                except Exception as e:
                    logger.error(f"fail to execute trouble shooter {shooter.__class__}: {e}")
                    return NoAction()

        return NoAction()
