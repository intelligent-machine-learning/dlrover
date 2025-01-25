from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction


class TroubleShooter(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def trouble_shoot(self, action: DiagnosisAction) -> DiagnosisAction:
        pass
    