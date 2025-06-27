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
import time
from typing import List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.enums import FailoverLevel
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.common.job_context import RestartInfo
from dlrover.python.unified.master.failover import FailoverCoordinator


class MPMDFailoverCoordinator(FailoverCoordinator):
    """
    To coordinate job management when failure happens in MPMD mode.
    """

    def handle_failures(self, failures: List[FailureDesc]):
        # process 1st failure each time if there is multi failures
        failure = failures[0]

        with self._lock:
            if self._is_failover_stage():
                logger.info(
                    f"Ignore failure: {failure} for already in "
                    "failover stage."
                )
                return

            logger.info(f"Handle failure: {failure}")
            level = self._get_failover_level(failure)
            role = failure.workload_role
            if self._is_failure_exceeded_limit(failure):
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

            self._set_failover_stage()

            if level == FailoverLevel.PARTIAL:
                self._do_partial_failover(failure)
            elif level == FailoverLevel.IGNORE:
                self._ignore_failover(failure)
            else:
                self._do_global_failover(failure)

            # update restart info
            if failure.is_workload_failure():
                self._job_context.add_restart_info(
                    failure.workload_role,
                    RestartInfo(restart_time=failure.failure_time),
                )
            elif failure.is_trainer_failure():
                self._job_context.add_trainer_restart_info(
                    RestartInfo(restart_time=failure.failure_time)
                )
            else:
                self._job_context.add_master_restart_info(
                    RestartInfo(restart_time=failure.failure_time)
                )

            self._reset_failover_stage()

    def _is_failure_exceeded_limit(self, failure: FailureDesc) -> bool:
        if failure.is_workload_failure():
            role = failure.workload_role
            return len(
                self.context.get_workload_restart_info(role)
            ) >= self.context.job_config.get_workload_max_restart(role)
        elif failure.is_trainer_failure():
            return (
                len(self.context.get_trainer_restart_info())
                >= self.context.job_config.trainer_max_restart
            )
        else:
            return (
                len(self.context.get_master_restart_info())
                >= self.context.job_config.master_max_restart
            )

    def _get_failover_level(self, failure: FailureDesc):
        level = failure.failure_level
        if level == 0:
            return FailoverLevel.IGNORE
        elif level == 1:
            return FailoverLevel.PARTIAL
        else:
            if level == -1:
                # TODO
                pass
            return FailoverLevel.GLOBAL

    def _is_job_restart_exceeded_limit(self) -> bool:
        return (
            len(self.context.get_job_restart_info())
            >= self.context.job_config.job_max_restart
        )

    def _do_global_failover(self, failure: FailureDesc):
        logger.info("Trigger global failover procedure.")
        if self._is_job_restart_exceeded_limit():
            logger.error(
                "Job restart exceed limit: "
                f"{self.context.job_config.job_max_restart}"
            )
            self._abort_job()
            return

        start = int(time.time())

        self.job_manager.stop_job()

        wait_interval = DLMasterConstant.GLOBAL_FAILOVER_INTERVAL
        for i in range(wait_interval):
            logger.info(
                f"Global failover will restart job in {wait_interval - i}s"
            )
            time.sleep(1)

        self.job_manager.start_job()
        self._save_context()
        logger.info(
            f"Global failover procedure cost {time.time() - start:.2f}s"
        )

        self._job_context.add_job_restart(start, True)

    def _do_partial_failover(self, failure: FailureDesc):
        logger.info("Trigger partial failover procedure.")
        # TODO
        pass
