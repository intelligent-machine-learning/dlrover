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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from dlrover.python.common.constants import (
    EventReportConstants,
    PreCheckStatus,
)
from dlrover.python.common.event.reporter import report_event
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.util.function_util import TimeoutException
from dlrover.python.util.time_util import get_pending_timeout

job_ctx = get_job_context()


@dataclass
class PreCheckResult:
    # The default success result is 0. The other result code(>0) should be
    # defined by different pre-check operator it's self.
    result: int = 0

    # The simple description info for the result.
    result_msg: str = ""

    # Abnormal nodes
    abnormal_nodes: List[Node] = field(default_factory=list)

    def is_success(self):
        return self.result == 0


class PreCheckOperator(ABC):
    @classmethod
    def get_retry_interval_secs(cls) -> int:
        """The retry interval seconds, can be overridden in subclasses."""
        return 5

    @abstractmethod
    def check(self, *args, **kwargs) -> PreCheckResult:
        """The abstraction of the main check procedure."""
        pass

    @abstractmethod
    def failed_actions(self, result: PreCheckResult) -> List[DiagnosisAction]:
        """The abstraction of the actions when operator check failed."""
        pass


class PreCheckManager:
    def __init__(self) -> None:
        self.operators: List[PreCheckOperator] = []
        self.status = PreCheckStatus.CHECKING

    def get_pre_check_timeout(self):
        return get_pending_timeout() + 600

    def do_check(self):
        if len(self.operators) == 0:
            report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_PRE_CHECK_DISABLE,
            )
            logger.info("No pre-check operators configured, skip pre-check.")
            self.status = PreCheckStatus.DISABLED
            return

        start = time.time()
        if self.status == PreCheckStatus.PASS:
            logger.info("Skip pre-check for the result is pass.")
            return

        logger.info(
            "Start training pre-check with "
            f"operators: {[op.__class__.__name__ for op in self.operators]} "
            f"under timeout: {self.get_pre_check_timeout()}s."
        )

        # 1. The all configured pre-check operators will be executed 1 by 1.
        # 2. If any operator check failed, the 'failed actions' will be
        # executed, and all the configured pre-check operators will be
        # executed once more after a 'waiting time' specified by the current
        # operator.
        # 3. There is no retry logic on each operator, because all the
        # operators must re-check if any failed action is executed.
        # 4. If any operator check fails and bypass is set to true, the
        # current result will be ignored, and the process will continue.
        # 5. If the there isn't any 'JobAbortion' during procedure, and the
        # pre-check procedure runs for a long time without finishing, which
        # will be considered as a flaw in the operator's execution. A warning
        # log will be triggered due to a timeout, and the result will be
        # marked as "pass."
        round = 0
        while True:
            logger.info(f"Pre-check round: {round}")
            for index, pre_check_op in enumerate(self.operators):
                bypass = _dlrover_context.is_pre_check_operator_bypass(
                    pre_check_op
                )
                success = self._do_check(
                    pre_check_op,
                    index,
                    ignore_failure=bypass,
                )
                if not success:
                    round += 1
                    continue
            break  # success

        self.status = PreCheckStatus.PASS
        report_event(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_PRE_CHECK_PASS,
        )
        logger.info(
            f"Training pre-check complete, cost:{time.time() - start:.2f}s."
        )

    def _do_check(
        self, op: PreCheckOperator, index: int, ignore_failure: bool = False
    ) -> bool:
        start = time.time()
        name = op.__class__.__name__
        try:
            result = op.check(job_args=self._job_args)
            logger.info(
                f"{name}({index}) done checking, "
                f"cost: {time.time() - start:.2f}s, "
                f"result: {result}"
            )
            if result.is_success():
                return True
            if ignore_failure:
                logger.warning(
                    f"{name}({index}) execute failed, "
                    "but pre-check pass due to bypass is enabled "
                )
                return True

            # go failed actions if check not passed
            actions = op.failed_actions(result)
            job_ctx.enqueue_actions(actions)
            wait_secs = op.get_retry_interval_secs()
            logger.info(
                f"{name} execute failed actions: {actions} and wait for {wait_secs}s"
            )
            time.sleep(wait_secs)
            return False  # next round
        except TimeoutException:
            report_event(
                EventReportConstants.TYPE_WARN,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_PRE_CHECK_TIMEOUT,
                name,
            )
            raise
        except Exception as e:
            report_event(
                EventReportConstants.TYPE_WARN,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_PRE_CHECK_ERROR,
                name,
            )
            logger.error(
                f"{name} got unexpected error: {e}",
                exc_info=True,
            )
            return True  # Let it pass
