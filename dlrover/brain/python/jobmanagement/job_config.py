# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Dict

from dlrover.brain.python.common.job import JobMeta


class JobConfigValues:
    def __init__(self):
        pass

    def convert_to_dict(self) -> Dict[str, str]:
        return {}


class JobConfigScope:
    def __init__(
        self,
    ):
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
        return self._include_scope.inscope(
            job
        ) and not self._exclude_scope.inscope(job)

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
