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
from typing import List

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.common.job_context import RestartInfo
from dlrover.python.unified.master.failover import FailoverCoordinator

FAILURE_TYPE_KEY = "FAILURE_TYPE"


class ElasticFailoverCoordinator(FailoverCoordinator):
    """
    To coordinate job management when failure happens in elastic mode.
    """

    def handle_failures(self, failures: List[FailureDesc]):
        logger.info(f"ElasticFailoverCoordinator handle failures: {failures}")
        for failure in failures:
            if self._is_agent_failure(failure):
                self._handle_agent_failure(failure)
            else:
                self._handle_process_failure(failure)

    def _is_agent_failure(self, failure: FailureDesc):
        if (
            failure.extra_info.get("FAILURE_TYPE")
            == TrainingExceptionLevel.NODE_ERROR
        ):
            return True
        return False

    def _handle_agent_failure(self, failure: FailureDesc):
        logger.info(
            f"ElasticFailoverCoordinator handle agent failure: {failure}"
        )
        role = failure.workload_role
        if self._is_agent_failure_exceeded_limit(failure):
            logger.error(
                f"Failure exceed limit, master limit: "
                f"{len(self.context.get_master_restart_info())}/"
                f"{self.context.job_config.master_max_restart}, "
                f"workload({role}) limit: "
                f"{len(self.context.get_workload_restart_info(role))}/"
                f"{self.context.job_config.get_workload_max_restart(role)}"
            )
            self._abort_job()
            return

        # do failover
        wl_name = failure.workload_name
        try:
            self.job_manager.re_execute(wl_name)
        except Exception as e:
            logger.error(
                f"Failed to execute on target: {wl_name} for "
                f"failover due to unexpected error: {e}"
            )

        self._job_context.add_restart_info(
            failure.workload_role,
            RestartInfo(restart_time=failure.failure_time),
        )

    def _handle_process_failure(self, failure: FailureDesc):
        logger.info(
            f"Process failure: {failure} is handled by "
            "elastic agent automated."
        )

    def _is_agent_failure_exceeded_limit(self, failure: FailureDesc) -> bool:
        role = failure.workload_role
        return len(
            self.context.get_workload_restart_info(role)
        ) >= self.context.job_config.get_workload_max_restart(role)
