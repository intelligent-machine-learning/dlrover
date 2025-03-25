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

from dlrover.python.rl.master.execution.scheduling_strategy import (
    SimpleOrderedStrategy,
)
from dlrover.python.rl.master.job_manager import JobManager
from dlrover.python.rl.tests.master.base import BaseMasterTest


class JobManagerTest(BaseMasterTest):
    def test_basic(self):
        job_manager = JobManager()
        self.assertTrue(
            isinstance(
                job_manager._get_scheduling_strategy(), SimpleOrderedStrategy
            )
        )

        job_manager._executor.execute = MagicMock(return_value=None)
        job_manager.start()
        job_manager.stop()
