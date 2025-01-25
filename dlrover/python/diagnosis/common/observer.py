from abc import ABCMeta, abstractmethod
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction


class Observer(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def observe(self) -> DiagnosisAction:
        pass
