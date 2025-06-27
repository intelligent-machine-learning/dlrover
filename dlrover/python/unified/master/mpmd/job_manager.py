#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.job_manager import JobManager
from dlrover.python.unified.master.mpmd.executor import MPMDTrainerExecutor


class MPMDJobManager(JobManager):
    def get_executor(self):
        return MPMDTrainerExecutor(self._execution_graph)

    def has_job_error(self):
        return self._executor.is_trainer_error()

    def gen_failures_by_error(self) -> List[FailureDesc]:
        if self.has_job_error():
            return [
                FailureDesc(
                    failure_obj="TRAINER",
                    failure_level=-1,
                    reason=self._executor.get_error(),
                )
            ]
        return []
