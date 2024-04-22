from abc import ABCMeta, abstractmethod


class Inference(object):
    def __init__(self):
        self.name = ""
        self.attribute = ""
        self.description = ""


class InferenceOperator(metaclass=ABCMeta):
    """
    InferenceOperator is used to infer the root cause of problems
    """
    def __init__(self):
        pass

    @abstractmethod
    def infer(self, inferences: list[Inference]) -> list[Inference]:
        pass


class InferenceChain:
    def __init__(self):
        pass

    def diagnose(self):
        pass


