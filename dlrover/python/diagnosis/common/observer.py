from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)


class Observer(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def observe(self) -> DiagnosisAction:
        return NoAction()

