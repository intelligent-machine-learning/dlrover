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
from unittest.mock import MagicMock

from dlrover.python.unified.master.graph import DLExecutionGraph
from dlrover.python.unified.master.mpmd.executor import MPMDTrainerExecutor
from dlrover.python.unified.tests.master.base import BaseMasterTest


class ExecutorTest(BaseMasterTest):
    def test_execute(self):
        graph = DLExecutionGraph(self._job_context.dl_context)
        executor = MPMDTrainerExecutor(graph)
        self.assertIsNotNone(executor.graph)

        executor.create_workloads = MagicMock(return_value=None)
        executor.execute()
