from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)


class TroubleShooter(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def trouble_shoot(self, action: DiagnosisAction) -> DiagnosisAction:
        return NoAction()

    @abstractmethod
    def is_compatible(self, action: DiagnosisAction) -> bool:
        return False
    