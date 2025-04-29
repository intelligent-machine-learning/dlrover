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
from typing import List, Tuple, Union

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEventType,
    NodeStatus,
    NodeType,
    PendingTimeoutStrategyType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisErrorConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    JobAbortionAction,
    NoAction,
    NodeAction,
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

    @abstractmethod
    def check(self, *args, **kwargs) -> PreCheckResult:
        """The abstraction of the main check procedure."""
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
    SCHEDULING_FAILED_MSG = "SCHEDULING_FAILED"
    PENDING_TIMEOUT_MSG = "PRE_CHECK_PENDING_TIMEOUT"
    PENDING_WAIT_MSG = "PRE_CHECK_PENDING_WAIT"

    @classmethod
    def get_retry_interval_secs(cls) -> int:
        return 15

    @classmethod
    def check_allreduce_job_pending(
        cls, cur_nodes, timeout, strategy
    ) -> Tuple[bool, Union[None, Node]]:
        """Return: ({has_pending}, {pending_timeout_node})"""

        logger.debug(f"Check current nodes: {cur_nodes}")
        pending_workers: List[Node] = []
        now = time.time()
        for node in cur_nodes:
            if node is None or node.is_released:
                continue
            if node.status in [NodeStatus.PENDING, NodeStatus.INITIAL]:
                pending_workers.append(node)

        if pending_workers:
            # find none create time element 1st
            first_pending_wk = next(
                (
                    pending_worker
                    for pending_worker in pending_workers
                    if not pending_worker.create_time
                ),
                None,
            )
            # then the min create time element if no none creat time element
            if not first_pending_wk:
                first_pending_wk = min(
                    pending_workers,
                    key=lambda x: x.create_time,
                )  # type: ignore

            if (
                first_pending_wk
                and first_pending_wk.create_time
                and now - first_pending_wk.create_time.timestamp() > timeout
            ):
                logger.warning(
                    f"Node {first_pending_wk.name} "
                    f"exceeded pending timeout: {timeout}s, "
                    f"job-type: {DistributionStrategy.ALLREDUCE}, "
                    f"strategy: {strategy}, "
                    f"pending workers(size:{len(pending_workers)})"
                    f": {pending_workers}, "
                )
                return True, first_pending_wk
            logger.debug(f"Exist pending worker: {first_pending_wk.name}")
            return True, None
        else:
            logger.info("No pending workers.")
            return False, None

    @classmethod
    def check_ps_job_pending(
        cls, cur_nodes, timeout, strategy
    ) -> Tuple[bool, Union[None, Node]]:
        """Return: ({has_pending}, {pending_timeout_node})"""

        logger.debug(f"Check current nodes: {cur_nodes}")
        pending_ps: List[Node] = []
        pending_workers: List[Node] = []
        now = time.time()
        for node in cur_nodes:
            if node is None or node.is_released:
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
                return True, first_pending_ps

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
                    return False, None

                if (
                    pending_worker_0.create_time
                    and now - pending_worker_0.create_time.timestamp()
                    > timeout
                ):
                    logger.warning(
                        f"Node {pending_worker_0.name} "
                        f"exceeded pending timeout: {timeout}s, "
                        f"job-type: {DistributionStrategy.PS}, "
                        f"strategy: {strategy}, "
                        f"pending workers(size:{len(pending_workers)})"
                        f": {pending_workers}."
                    )
                    return True, pending_worker_0
                return True, None
            else:
                first_pending_wk = min(
                    pending_workers, key=lambda x: x.create_time
                )  # type: ignore
                if (
                    first_pending_wk
                    and first_pending_wk.create_time
                    and now - first_pending_wk.create_time.timestamp()
                    > timeout
                ):
                    logger.warning(
                        f"Node {first_pending_wk.name} "
                        f"exceeded pending timeout: {timeout}s, "
                        f"job-type: {DistributionStrategy.PS}, "
                        f"strategy: {strategy}, "
                        f"pending workers(size:{len(pending_workers)})"
                        f": {pending_workers}, "
                    )
                    return True, first_pending_wk
                return True, None
        logger.info("No pending ps or workers.")
        return False, None

    @classmethod
    def wait_scheduling_started(cls, wait_time=10, timeout=300):
        start = time.time()
        while True:
            if time.time() - start > timeout:
                logger.warning(
                    f"Scheduling hasn't started for over {timeout}s."
                )
                return False

            has_started = False
            job_nodes = job_ctx.job_nodes()
            for _, nodes in job_nodes.items():
                for _, node in nodes.items():
                    if node.create_time:
                        has_started = True
            if has_started:
                return True
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
        logger.info(
            f"{self.__class__.__name__} do checking with "
            f"job_type: {job_type}, "
            f"strategy: {strategy}, "
            f"timeout: {timeout}"
        )

        if timeout <= 0 or strategy == PendingTimeoutStrategyType.SKIP:
            msg = (
                f"Skip {self.__class__.__name__} for "
                "'skip' pending timeout strategy."
            )
            logger.info(msg)
            return PreCheckResult(result_msg=msg)

        if not self.wait_scheduling_started():
            logger.warning("All nodes pending when scheduling.")
            return PreCheckResult(
                result=1,
                result_msg=SchedulingPreCheckOperator.SCHEDULING_FAILED_MSG,
                abnormal_nodes=[],
            )

        round = 0
        while True:
            logger.info(f"Scheduling pre-check round: {round}")
            if job_type == DistributionStrategy.ALLREDUCE:
                cur_nodes = list(
                    job_ctx.job_nodes_by_type(NodeType.WORKER).values()
                )
                pending_result = self.check_allreduce_job_pending(
                    cur_nodes, timeout, strategy
                )
            elif job_type == DistributionStrategy.PS:
                ps_nodes = list(
                    job_ctx.job_nodes_by_type(NodeType.PS).values()
                )
                worker_nodes = list(
                    job_ctx.job_nodes_by_type(NodeType.WORKER).values()
                )
                pending_result = self.check_ps_job_pending(
                    ps_nodes + worker_nodes, timeout, strategy
                )
            else:
                msg = f"Skip {self.__class__.__name__} for {job_type}."
                logger.warning(msg)
                return PreCheckResult(result_msg=msg)

            if pending_result[0]:
                # has pending node
                if pending_result[1]:
                    # has pending node timeout
                    return PreCheckResult(
                        result=1,
                        result_msg=SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG,  # noqa: E501
                        abnormal_nodes=[pending_result[1]],
                    )
                else:
                    # has pending node, wait for a while and try checking again
                    time.sleep(self.get_retry_interval_secs())
                    round += 1
                    continue
            else:
                # no pending node
                return PreCheckResult()

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        result_msg = str(kwargs.get("result_msg"))
        abnormal_nodes = kwargs.get("abnormal_nodes")
        msg = result_msg
        if (
            result_msg == SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG
            and isinstance(abnormal_nodes, list)
        ):
            msg = result_msg + ":" + str(abnormal_nodes[0].id)
        return [
            JobAbortionAction(
                reason=SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG, msg=msg
            )
        ]


class ConnectionPreCheckOperator(PreCheckOperator):
    """
    This operator will check whether all the target workers can establish
    connection with master.

    Notice:
    1) This operator is available for all-reduce(torch) job only for now.
    2) Can not be used independently, and must be used after
       'SchedulingPreCheckOperator'.
    """

    CONN_CHECK_FAILED_MSG = "CONNECTION_CHECK_FAILED"

    @classmethod
    def get_retry_interval_secs(cls) -> int:
        return 60  # need to wait node fault tolerance

    def _get_check_retry_times(self):
        """Can be overridden in subclass."""
        return 15

    def _get_check_retry_interval(self):
        """Can be overridden in subclass."""
        return 60

    def check(self, *args, **kwargs) -> PreCheckResult:
        retry_times = self._get_check_retry_times()
        each_retry_interval = self._get_check_retry_interval()
        abnormal_nodes = []

        job_nodes = job_ctx.job_nodes()

        # use a retry here
        for i in range(retry_times):
            for _, nodes in job_nodes.items():
                for _, node in nodes.items():
                    if (
                        node.status == NodeStatus.RUNNING
                        and node.reported_status[0]
                        != NodeEventType.WAIT_PRE_CHECK
                    ):
                        logger.debug(
                            f"Node {node.id} failed connection check, "
                            f"retry time: {i}."
                        )
                        abnormal_nodes.append(node)

            if abnormal_nodes:
                # with connection issue
                if i + 1 == retry_times:
                    # retry out of limit
                    break
                else:
                    logger.info(
                        f"Got {len(abnormal_nodes)} nodes with "
                        f"connection issue for {i}/{retry_times}, "
                        f"wait {each_retry_interval}s for next retry"
                    )
                    abnormal_nodes.clear()
                    time.sleep(each_retry_interval)
            else:
                # no connection issue
                break

        if abnormal_nodes:
            return PreCheckResult(
                result=1,
                result_msg=ConnectionPreCheckOperator.CONN_CHECK_FAILED_MSG,
                abnormal_nodes=abnormal_nodes,
            )

        return PreCheckResult()

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        abnormal_nodes = kwargs.get("abnormal_nodes")
        failed_actions: List[DiagnosisAction] = []
        if not abnormal_nodes or not isinstance(abnormal_nodes, List):
            logger.warning("No valid abnormal nodes for failed actions.")
            return failed_actions

        for node in abnormal_nodes:
            failed_actions.append(
                NodeAction(
                    node_status=DiagnosisErrorConstant.PRE_CHECK_FAILED,
                    reason=ConnectionPreCheckOperator.CONN_CHECK_FAILED_MSG,
                    node_id=node.id,
                    node_type=node.type,
                    instance=DiagnosisConstant.MASTER_INSTANCE,
                    action_type=DiagnosisActionType.MASTER_RELAUNCH_WORKER,
                )
            )
        return failed_actions
