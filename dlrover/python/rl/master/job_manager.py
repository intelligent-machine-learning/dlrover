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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.context import get_job_context
from dlrover.python.rl.common.enums import SchedulingStrategyType
from dlrover.python.rl.master.execution.executor import Executor
from dlrover.python.rl.master.execution.graph import RLExecutionGraph
from dlrover.python.rl.master.execution.scheduling_strategy import (
    SimpleStrategy,
)


class JobManager(object):
    def __init__(self):
        self._job_ctx = get_job_context()

        self._execution_graph = RLExecutionGraph(self._job_ctx.rl_context)
        self._executor = Executor(
            self._execution_graph, self._get_scheduling_strategy()
        )

    def _get_scheduling_strategy(self):
        strategy_type = self._job_ctx.job_config.scheduling_strategy_type
        if strategy_type == SchedulingStrategyType.SIMPLE:
            return SimpleStrategy()
        else:
            raise NotImplementedError()

    def start_job(self):
        logger.info("Starting job manager.")
        self._executor.execute()

    def is_job_finished(self):
        return self._executor.is_trainer_finished()

    def stop_job(self):
        self._executor.cleanup()
