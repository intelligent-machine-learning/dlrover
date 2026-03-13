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

from typing import List, Optional
from dlrover.brain.python.jobmanagement.job_config import (
    JobConfig,
    JobConfigValues,
)
from dlrover.brain.python.common.job import JobMeta


class JobConfigManager:
    """
    Manages job-specific configurations. For a given job, JobConfigManager
    will retrieve the configuration that matches this job. If none configuration
    matched, it returns None.
    """

    def __init__(self):
        self._configs: List[JobConfig] = []

    def get_job_config(self, job: JobMeta) -> Optional[JobConfigValues]:
        for config in self._configs:
            if config.in_scope(job):
                return config.config_values
        return None
