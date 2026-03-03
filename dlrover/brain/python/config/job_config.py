from typing import List, Optional, Dict

from dlrover.brain.python.common.job import JobMeta


class JobConfigValues:
    def __init__(self):
        pass

    def convert_to_dict(self) -> Dict[str, str]:
        return {}


class JobConfigScope:
    def __init__(self,):
        pass

    def inscope(self, job: JobMeta) -> bool:
        return True


class JobConfig:
    def __init__(self):
        self._name = ""
        self._include_scope: JobConfigScope = JobConfigScope()
        self._exclude_scope: JobConfigScope = JobConfigScope()
        self._config_values: JobConfigValues = JobConfigValues()

    def inscope(self, job: JobMeta) -> bool:
        return self._include_scope.inscope(job) and not self._exclude_scope.inscope(job)

    @property
    def config_values(self) -> JobConfigValues:
        return self._config_values


class JobConfigManager:
    def __init__(self):
        self._configs: List[JobConfig] = []

    def get_job_config(self, job: JobMeta) -> Optional[JobConfigValues]:
        for config in self._configs:
            if config.inscope(job):
                return config.config_values
        return None
