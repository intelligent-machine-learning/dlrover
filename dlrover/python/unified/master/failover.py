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
import threading
from abc import ABC, abstractmethod
from typing import List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLJobExitReason
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.common.job_context import get_job_context
from dlrover.python.unified.master.job_manager import JobManager


class FailoverCoordinator(ABC):
    """
    To coordinate job management when failure happens.
    """

    def __init__(self, job_manager, save_context_callback, exit_job_callback):
        self._job_manager: JobManager = job_manager
        self._save_context_callback = save_context_callback
        self._exit_job_callback = exit_job_callback

        self._job_context = get_job_context()

        self._lock = threading.Lock()

    @property
    def context(self):
        return self._job_context

    @property
    def job_manager(self):
        return self._job_manager

    def _is_failover_stage(self):
        self._job_context.is_in_failover()

    def _set_failover_stage(self):
        self._job_context.set_in_failover_stage()

    def _reset_failover_stage(self):
        self._job_context.set_job_stage(JobStage.RUNNING)

    @abstractmethod
    def handle_failures(self, failures: List[FailureDesc]):
        """Core processing for failure handling."""

    def _ignore_failover(self, failure: FailureDesc):
        logger.info(f"Ignore failover for failure: {failure}")

    def _save_context(self):
        logger.info("Save context after failover.")
        self._save_context_callback()

    def _abort_job(self):
        logger.info("Abort job for failover can no longer proceed.")
        self._exit_job_callback(
            stage=JobStage.ERROR,
            forced=True,
            reason=DLJobExitReason.FAILOVER_OUT_OF_LIMIT,
        )
