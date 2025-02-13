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
from typing import List, Union

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeStatus,
    NodeType,
    PendingTimeoutStrategyType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    JobAbortionAction,
    NoAction,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.util.time_util import get_pending_timeout

_dlrover_ctx = Context.singleton_instance()
job_ctx = get_job_context()


@dataclass
class PreCheckResult(object):
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

    @classmethod
    def get_retry_times(cls) -> int:
        """
        The limited retry times, can be overridden in subclasses. For most
        pre-check, the retry value should > 1(at least once retry).

        The failed action will be executed if result still not ok after
        several retry times.
        """
        return 3

    @abstractmethod
    def check(self, *args, **kwargs) -> PreCheckResult:
        """The abstraction of the main check procedure."""
        pass

    @abstractmethod
    def recover_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        """The abstraction of the procedure actions if check failed."""
        pass

    @abstractmethod
    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        """The abstraction of the actions when operator check failed."""
        pass


class NoPreCheckOperator(PreCheckOperator):
    def check(self, *args, **kwargs):
        return PreCheckResult()

    def recover_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]


class SchedulingPreCheckOperator(PreCheckOperator):
    PENDING_TIMEOUT_MSG = "PRE_CHECK_PENDING_TIMEOUT"

    @classmethod
    def get_retry_interval_secs(cls) -> int:
        return 60

    @classmethod
    def get_retry_times(cls) -> int:
        timeout = get_pending_timeout()
        if timeout <= 0:
            return 1
        else:
            return int(timeout / cls.get_retry_interval_secs() + 1)

    @classmethod
    def check_allreduce_job_pending(
        cls, cur_nodes, timeout, strategy
    ) -> Union[None, Node]:
        pending_workers: List[Node] = []
        now = time.time()
        for node in cur_nodes:
            if node is None or node.is_released or node.create_time is None:
                continue
            if node.status in [NodeStatus.PENDING, NodeStatus.INITIAL]:
                pending_workers.append(node)

        if pending_workers:
            first_pending_wk = min(
                pending_workers,
                key=lambda x: x.create_time,
            )  # type: ignore
            if not first_pending_wk:  # type: ignore
                logger.info("No pending workers.")
                return None

            pending_time = (
                first_pending_wk.create_time.timestamp()  # type: ignore
            )
            if now - pending_time > timeout:
                logger.warning(
                    f"Node {first_pending_wk.name} "
                    f"exceeded pending timeout: {timeout}s, "
                    f"job-type: {DistributionStrategy.ALLREDUCE}, "
                    f"strategy: {strategy}, "
                    f"pending workers(size:{len(pending_workers)})"
                    f": {pending_workers}, "
                )
                return first_pending_wk

        logger.info("No pending workers.")
        return None

    @classmethod
    def check_ps_job_pending(
        cls, cur_nodes, timeout, strategy
    ) -> Union[None, Node]:
        pending_ps: List[Node] = []
        pending_workers: List[Node] = []
        now = time.time()
        for node in cur_nodes:
            if node is None or node.is_released or node.create_time is None:
                continue
            if node.status in [NodeStatus.PENDING, NodeStatus.INITIAL]:
                if node.type == NodeType.PS:
                    pending_ps.append(node)
                else:
                    pending_workers.append(node)

        # 1st: judge ps
        if pending_ps:
            first_pending_ps = min(
                pending_ps, key=lambda x: x.create_time  # type: ignore
            )
            if (
                first_pending_ps
                and first_pending_ps.create_time
                and (now - first_pending_ps.create_time.timestamp() > timeout)
            ):
                logger.warning(
                    f"Node {first_pending_ps.name} "
                    f"exceeded pending timeout: {timeout}s, "
                    f"job-type: {DistributionStrategy.PS}, "
                    f"strategy: {strategy}, "
                    f"pending ps(size:{len(pending_ps)})"
                    f": {pending_ps}, "
                )
                return first_pending_ps

        # 2nd: judge worker
        if pending_workers:
            if strategy == PendingTimeoutStrategyType.NECESSARY:
                # get worker 0
                pending_worker_0 = None
                for pending_worker in pending_workers:
                    if pending_worker.rank_index == 0:
                        pending_worker_0 = pending_worker
                        break
                if not pending_worker_0:  # type: ignore
                    logger.info("No pending worker(0).")
                    return None

                pending_time = (
                    pending_worker_0.create_time.timestamp()  # type: ignore
                )
                if now - pending_time > timeout:
                    logger.warning(
                        f"Node {pending_worker_0.name} "
                        f"exceeded pending timeout: {timeout}s, "
                        f"job-type: {DistributionStrategy.PS}, "
                        f"strategy: {strategy}, "
                        f"pending workers(size:{len(pending_workers)})"
                        f": {pending_workers}."
                    )
                    return pending_worker_0
            else:
                first_pending_wk = min(
                    pending_workers, key=lambda x: x.create_time
                )  # type: ignore
                if not first_pending_wk:  # type: ignore
                    logger.info("No pending workers.")
                    return None

                pending_time = (
                    first_pending_wk.create_time.timestamp()  # type: ignore
                )
                if now - pending_time > timeout:
                    logger.warning(
                        f"Node {first_pending_wk.name} "
                        f"exceeded pending timeout: {timeout}s, "
                        f"job-type: {DistributionStrategy.PS}, "
                        f"strategy: {strategy}, "
                        f"pending workers(size:{len(pending_workers)})"
                        f": {pending_workers}, "
                    )
                    return first_pending_wk

        logger.info("No pending ps or workers.")
        return None

    @classmethod
    def wait_scheduling_started(cls, wait_time=10, timeout=300):
        start = time.time()
        while True:
            if time.time() - start > timeout:
                logger.warning(
                    f"Scheduling hasn't started for over {timeout}s."
                )
                break

            has_started = False
            job_nodes = job_ctx.job_nodes()
            for _, nodes in job_nodes.items():
                for _, node in nodes.items():
                    if node.create_time:
                        has_started = True
            if has_started:
                break
            else:
                logger.info(
                    f"Scheduling hasn't started yet, wait {wait_time}s..."
                )
                time.sleep(wait_time)

    def check(self, *args, **kwargs):
        job_args: JobArgs = kwargs.get("job_args")
        job_type = job_args.distribution_strategy
        strategy = _dlrover_ctx.pending_fail_strategy
        timeout = get_pending_timeout()

        if timeout <= 0 or strategy == PendingTimeoutStrategyType.SKIP:
            msg = (
                f"Skip {self.__class__.__name__} for "
                "'skip' pending timeout strategy."
            )
            logger.info(msg)
            return PreCheckResult(result_msg=msg)

        self.wait_scheduling_started()

        if job_type == DistributionStrategy.ALLREDUCE:
            cur_nodes = list(
                job_ctx.job_nodes_by_type(NodeType.WORKER).values()
            )
            pending_node = self.check_allreduce_job_pending(
                cur_nodes, timeout, strategy
            )
        elif job_type == DistributionStrategy.PS:
            ps_nodes = list(job_ctx.job_nodes_by_type(NodeType.PS).values())
            worker_nodes = list(
                job_ctx.job_nodes_by_type(NodeType.WORKER).values()
            )
            pending_node = self.check_ps_job_pending(
                ps_nodes + worker_nodes, timeout, strategy
            )
        else:
            msg = f"Skip {self.__class__.__name__} for {job_type}."
            logger.warning(msg)
            return PreCheckResult(result_msg=msg)

        if pending_node:
            return PreCheckResult(
                result=1,
                result_msg=SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG,
                abnormal_nodes=[pending_node],
            )
        else:
            return PreCheckResult()

    def recover_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [JobAbortionAction()]
