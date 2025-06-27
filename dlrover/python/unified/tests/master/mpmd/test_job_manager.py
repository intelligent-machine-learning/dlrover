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
import os
from unittest.mock import MagicMock

from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.enums import SchedulingStrategyType
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.mpmd.job_manager import MPMDJobManager
from dlrover.python.unified.master.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)
from dlrover.python.unified.tests.master.base import BaseMasterTest


class JobManagerTest(BaseMasterTest):
    def setUp(self):
        super(JobManagerTest, self).setUp()
        os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"

    def tearDown(self):
        os.environ.clear()
        super(JobManagerTest, self).tearDown()

    def test_basic(self):
        job_manager = MPMDJobManager()
        self.assertTrue(
            isinstance(job_manager.get_scheduler(), GroupOrderedScheduler)
        )
        self.assertFalse(job_manager.has_job_error())
        self.assertFalse(job_manager.gen_failures_by_error())
        job_manager.gen_failures_by_error = MagicMock(
            return_value=[FailureDesc(failure_obj="Trainer", reason="test")]
        )
        self.assertEqual(job_manager.gen_failures_by_error()[0].reason, "test")

        job_manager._executor.execute = MagicMock(return_value=None)
        job_manager.start_job()
        job_manager.stop_job()

    def test_get_scheduler(self):
        job_manager = MPMDJobManager()
        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.SIMPLE
        )
        self.assertTrue(
            isinstance(job_manager.get_scheduler(), SimpleScheduler)
        )

        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.GROUP
        )
        self.assertTrue(
            isinstance(job_manager.get_scheduler(), GroupOrderedScheduler)
        )

        job_manager._job_ctx.dl_context.has_workload_group = MagicMock(
            return_value=True
        )
        self.assertTrue(
            isinstance(job_manager.get_scheduler(), GroupOrderedScheduler)
        )

        job_manager._get_scheduling_type_from_context = MagicMock(
            return_value=SchedulingStrategyType.AUTO
        )
        self.assertTrue(
            isinstance(job_manager.get_scheduler(), GroupOrderedScheduler)
        )
