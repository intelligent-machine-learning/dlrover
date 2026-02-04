from abc import ABC, abstractmethod

from dlrover.brain.python.common.optimize import JobMeta


class JobConfig:
    def __init__(self):
        pass


class JobConfigScope:
    def __init__(self,):
        pass

    def inscope(self, job: JobMeta) -> bool:
        return True


class JobConfigItem:
    def __init__(self):
        self._name = ""
        self._include_scope: JobConfigScope = JobConfigScope()
        self._exclude_scope: JobConfigScope = JobConfigScope()
        self._job_config: JobConfig = JobConfig()


class JobConfigReader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_config(self) -> JobConfigItem:
        pass

