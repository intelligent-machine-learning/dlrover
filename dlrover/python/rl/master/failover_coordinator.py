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
from dlrover.python.rl.common.enums import FailoverLevel
from dlrover.python.rl.common.failure import FailureDesc
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.job_manager import JobManager


class FailoverCoordinator(object):
    """
    To coordinate job management when failure happens.
    """

    def __init__(self, job_manager, save_context_callback):
        self._job_manager: JobManager = job_manager
        self._save_context_callback = save_context_callback
        self._job_context = get_job_context()

    def handle_failure(self, failure: FailureDesc):
        logger.info(f"Handle failure: {failure}")
        level = self._get_failover_level(failure)

        if level == FailoverLevel.PARTIAL:
            self._do_partial_failover(failure)
        elif level == FailoverLevel.IGNORE:
            self._ignore_failover(failure)
        else:
            self._do_global_failover(failure)

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

    def _do_global_failover(self, failure: FailureDesc):
        logger.info("Trigger global failover procedure.")
        self._job_manager.stop_job()
        self._job_manager.start_job()
        self._save_context()

    def _do_partial_failover(self, failure: FailureDesc):
        logger.info("Trigger partial failover procedure.")
        # TODO
        pass

    def _ignore_failover(self, failure: FailureDesc):
        logger.info(f"Ignore failover for failure: {failure}")

    def _save_context(self):
        logger.info("Save context after failover.")
        self._save_context_callback()
