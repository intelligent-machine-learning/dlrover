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
from dlrover.python.rl.common.enums import SchedulingStrategyType
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.execution.executor import Executor
from dlrover.python.rl.master.execution.graph import RLExecutionGraph
from dlrover.python.rl.master.execution.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)


class JobManager(object):
    def __init__(self):
        self._job_ctx = get_job_context()

        self._execution_graph = RLExecutionGraph(self._job_ctx.rl_context)
        self._job_ctx.set_execution_graph(self._execution_graph)

        self._scheduler = self._get_scheduler()
        self._executor = Executor(self._execution_graph)

    def _get_scheduling_type_from_context(self):
        return self._job_ctx.job_config.scheduling_strategy_type

    def _get_scheduler(self):
        strategy_type = self._get_scheduling_type_from_context()
        if strategy_type == SchedulingStrategyType.SIMPLE:
            logger.info("Use simple strategy for scheduling by specification.")
            return SimpleScheduler(self._execution_graph)
        elif strategy_type == SchedulingStrategyType.GROUP:
            if self._job_ctx.rl_context.has_workload_group():
                logger.info(
                    "Use group strategy for scheduling by specification."
                )
                return GroupOrderedScheduler(self._execution_graph)
            else:
                logger.info(
                    "Downgrade to simple strategy for scheduling because "
                    "workload group description is empty in rl-context."
                )
                return SimpleScheduler(self._execution_graph)
        else:
            # for auto type:
            # use group strategy if exits group in context,
            # or use simple strategy
            if self._job_ctx.rl_context.has_workload_group():
                logger.info("Use group strategy for scheduling by auto.")
                return GroupOrderedScheduler(self._execution_graph)
            else:
                logger.info("Use simple strategy for scheduling by auto.")
                return SimpleScheduler(self._execution_graph)

    def start_job(self):
        logger.info("Start job execution.")

        # create workloads
        self.create_workloads()

        # execute trainer
        self._executor.execute()

    def stop_job(self):
        # destroy all workloads
        self.destroy_workloads()

    def create_workloads(self):
        """Sync operation."""
        logger.info("Create all workloads.")

        self._scheduler.schedule()

    def destroy_workloads(self):
        """Sync operation."""

        logger.info("Destroy all workloads.")
        self._scheduler.cleanup()

    def is_job_finished(self):
        return self._executor.is_trainer_finished()
