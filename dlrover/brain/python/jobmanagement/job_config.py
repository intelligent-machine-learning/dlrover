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
    def __init__(self, configs: Optional[Dict[str, str]] = None):
        self._configs: Dict[str, str] = {}
        if configs is not None:
            self._configs = configs

    def get_config_values(self) -> Dict[str, str]:
        return self._configs


class JobConfigScope:
    def __init__(
        self,
        conds: Optional[Dict[str, List[str]]] = None,
    ):
        self._conds: Dict[str, List[str]] = {}
        if conds is not None:
            self._conds = conds

    def in_scope(self, job: JobMeta) -> bool:
        if "user" in self._conds and job.user not in self._conds["user"]:
            return False
        if (
            "namespace" in self._conds
            and job.namespace not in self._conds["namespace"]
        ):
            return False
        if (
            "cluster" in self._conds
            and job.cluster not in self._conds["cluster"]
        ):
            return False
        if "app" in self._conds and job.app not in self._conds["app"]:
            return False
        return True


class JobConfig:
    def __init__(
        self,
        include_scope: Optional[JobConfigScope] = None,
        exclude_scope: Optional[JobConfigScope] = None,
    ):
        self._name = ""
        self._include_scope: Optional[JobConfigScope] = include_scope
        self._exclude_scope: Optional[JobConfigScope] = exclude_scope
        self._config_values: JobConfigValues = JobConfigValues()

    def in_scope(self, job: JobMeta) -> bool:
        return (
            self._include_scope is None or self._include_scope.in_scope(job)
        ) and (
            self._exclude_scope is None
            or not self._exclude_scope.in_scope(job)
        )

    @property
    def config_values(self) -> JobConfigValues:
        return self._config_values
