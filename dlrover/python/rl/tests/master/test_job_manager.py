# Copyright 2025 The DLRover Authors. All rights reserved.
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
from unittest.mock import MagicMock

from dlrover.python.rl.common.enums import SchedulingStrategyType
from dlrover.python.rl.master.execution.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)
from dlrover.python.rl.master.job_manager import JobManager
from dlrover.python.rl.tests.master.base import BaseMasterTest


class JobManagerTest(BaseMasterTest):
    def test_basic(self):
        job_manager = JobManager()
        self.assertTrue(
            isinstance(job_manager._get_scheduler(), SimpleScheduler)
        )

        job_manager._executor.execute = MagicMock(return_value=None)
        job_manager.start_job()
        job_manager.stop()

    def test_get_scheduling_strategy(self):
        job_manager = JobManager()
        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.SIMPLE
        )
        self.assertTrue(
            isinstance(job_manager._get_scheduler(), SimpleScheduler)
        )

        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.GROUP
        )
        self.assertTrue(
            isinstance(job_manager._get_scheduler(), SimpleScheduler)
        )

        job_manager._job_ctx.rl_context.has_workload_group = MagicMock(
            return_value=True
        )
        self.assertTrue(
            isinstance(job_manager._get_scheduler(), GroupOrderedScheduler)
        )

        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.AUTO
        )
        self.assertTrue(
            isinstance(job_manager._get_scheduler(), GroupOrderedScheduler)
        )
