from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.resolver import Resolver
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
        self._resolvers: List[Resolver] = []
        self._lock = threading.Lock()

    def observe(self, problem: str, **kwargs) -> DiagnosisAction:
        with self._lock:
            observer = self._observers.get(problem, None)
        if observer is None:
            return NoAction()
        try:
            return observer.observe(**kwargs)
        except Exception as e:
            logger.error(f"fail to execute observer {observer.__class__}: {e}")
            return NoAction()

    def register_observer(self, name: str, observer: Observer):
        with self._lock:
            if len(name) == 0:
                return
            self._observers[name] = observer

    def register_resolver(self, resolver: Resolver):
        with self._lock:
            if resolver is None:
                return
            self._resolvers.append(resolver)

    def resolve(self, action: DiagnosisAction) -> DiagnosisAction:
        for resolver in self._resolvers:
            if resolver.is_compatible(action):
                try:
                    return resolver.resolve(action)
                except Exception as e:
                    logger.error(f"fail to execute trouble shooter {resolver.__class__}: {e}")
                    return NoAction()

        return NoAction()
