from abc import ABCMeta, abstractmethod


class Scaler(metaclass=ABCMeta):
    """Scaler is to call cluster scheduler to scale up/down nodes of a job.
    Attributes:
        job_name: string, the name of job.
    """
    def __init__(self, job_name):
        self._job_name = job_name

    @abstractmethod
    def scale(self, resource_plan):
        pass
