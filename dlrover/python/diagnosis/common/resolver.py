from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)


class Resolver(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def resolve(self, action: DiagnosisAction) -> DiagnosisAction:
        return NoAction()

    @abstractmethod
    def is_compatible(self, action: DiagnosisAction) -> bool:
        return False
    