# Copyright 2025 The EasyDL Authors. All rights reserved.
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
import time
from unittest.mock import MagicMock

from dlrover.python.rl.common.failure import FailureDesc
from dlrover.python.rl.master.failover_coordinator import FailoverCoordinator
from dlrover.python.rl.master.job_manager import JobManager
from dlrover.python.rl.tests.master.base import BaseMasterTest


class FailoverCoordinatorTest(BaseMasterTest):
    def test_handle_failure(self):
        job_manager = JobManager()
        job_manager.start_job = MagicMock(return_value=None)
        job_manager.stop_job = MagicMock(return_value=None)

        def callback():
            return

        fc = FailoverCoordinator(job_manager, callback, callback)

        desc = FailureDesc(
            workload_name="test",
            workload_role="ACTOR",
            failure_time=int(time.time()),
            failure_level=2,
            reason="unknown",
        )

        fc.handle_failure(desc)
        job_manager.start_job.assert_called_once()
        job_manager.stop_job.assert_called_once()

        desc = FailureDesc(
            failure_obj="MASTER",
            failure_time=int(time.time()),
            failure_level=0,
            reason="unknown",
        )
        fc.handle_failure(desc)
        job_manager.start_job.assert_called_once()
        job_manager.stop_job.assert_called_once()
