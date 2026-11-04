# Copyright 2026 The DLRover Authors. All rights reserved.
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

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisDataType,
    DiagnosisErrorConstant,
    DiagnosisState,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NodeAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)


DEFAULT_CORRELATION_WINDOW_SECONDS = 300
DEFAULT_PENDING_ACTION_TIMEOUT_SECONDS = 60
DEFAULT_ACTION_EXPIRY_SECONDS = 300
DEFAULT_ROUND_RETENTION_SECONDS = 1800
DEFAULT_MAX_ROUNDS = 1024
DEFAULT_REPORT_BATCH_SIZE = 100


@dataclass
class JobPerformance:
    """Master-side decision and action state for one performance episode."""

    performance_id: str
    correlation_key: str
    state: str
    job_uid: str
    run_id: str
    rdzv_epoch: int
    config_fingerprint: str
    observed_at: int
    created_at: int
    updated_at: int
    issue_types: List[str] = field(default_factory=list)
    source_report_ids: List[str] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    action_group_id: str = ""
    planned_action_ids: List[str] = field(default_factory=list)
    queued_action_ids: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self):
        return {k.lstrip("_"): v for k, v in self.__dict__.items()}


class JobPerformanceDiagnostician(Diagnostician):
    """Consume worker performance reports and decide job-level actions."""

    def __init__(
        self,
        data_manager: DiagnosisDataManager,
        job_args=None,
        correlation_window_seconds=DEFAULT_CORRELATION_WINDOW_SECONDS,
        pending_action_timeout_seconds=(
            DEFAULT_PENDING_ACTION_TIMEOUT_SECONDS
        ),
        action_expiry_seconds=DEFAULT_ACTION_EXPIRY_SECONDS,
        round_retention_seconds=DEFAULT_ROUND_RETENTION_SECONDS,
        max_rounds=DEFAULT_MAX_ROUNDS,
        report_batch_size=DEFAULT_REPORT_BATCH_SIZE,
    ):
        super().__init__(job_args)
        self._data_manager = data_manager
        self._last_processed_sequence_id = 0
        self._correlation_window_seconds = max(0, correlation_window_seconds)
        self._pending_action_timeout_seconds = max(
            1, pending_action_timeout_seconds
        )
        self._action_expiry_seconds = max(1, action_expiry_seconds)
        self._round_retention_seconds = max(
            self._correlation_window_seconds, round_retention_seconds
        )
        self._max_rounds = max(1, max_rounds)
        self._report_batch_size = max(1, report_batch_size)
        self._rounds: Dict[str, JobPerformance] = {}
        self._round_order: Deque[str] = deque()
        self._ready_rounds: Deque[str] = deque()

    @property
    def last_processed_sequence_id(self):
        return self._last_processed_sequence_id

    @property
    def rounds(self):
        return dict(self._rounds)

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        logger.info(
            "[PERF-DIAG][master] Performance check started: "
            "paused=%s, data_manager=%s, last_sequence_id=%s, "
            "rounds=%d, ready_rounds=%d.",
            kwargs.get("observing_paused"),
            self._data_manager is not None,
            self._last_processed_sequence_id,
            len(self._rounds),
            len(self._ready_rounds),
        )
        if kwargs.get("observing_paused") or self._data_manager is None:
            logger.info(
                "[PERF-DIAG][master] Performance check skipped: "
                "observing is paused or data manager is unavailable."
            )
            return None

        now = int(time.time())
        self._prune_rounds(now)
        reports = self._data_manager.get_data(
            DiagnosisDataType.PERFORMANCE_ISSUE,
            after_sequence_id=self._last_processed_sequence_id,
            limit=self._report_batch_size,
        )
        logger.info(
            "[PERF-DIAG][master] Fetched performance reports: "
            "count=%d, sequence_ids=%s.",
            len(reports),
            [report.sequence_id for report in reports],
        )
        for report in reports:
            logger.info(
                "[PERF-DIAG][master] Processing report: "
                "sequence_id=%s, report_id=%s, node=%s, rank=%s, "
                "issue=%s, metric=%s, value=%s, baseline=%s, "
                "threshold=%s, samples=%s, observed_at=%s.",
                report.sequence_id,
                report.report_id,
                report.node_id,
                report.node_rank,
                report.issue_type,
                report.metric_name,
                report.metric_value,
                report.baseline_value,
                report.threshold_value,
                report.sample_count,
                self._report_observed_at(report),
            )
            self._process_report(report, kwargs)
            self._last_processed_sequence_id = report.sequence_id

        self._expire_pending_rounds(now)
        while self._ready_rounds:
            performance_id = self._ready_rounds[0]
            performance = self._rounds.get(performance_id)
            if (
                performance is None
                or performance.state != DiagnosisState.ISSUE
            ):
                self._ready_rounds.popleft()
                continue
            logger.info(
                "[PERF-DIAG][master] Performance issue ready: "
                "performance_id=%s, reports=%s, nodes=%s, issues=%s.",
                performance_id,
                performance.source_report_ids,
                performance.source_node_ids,
                performance.issue_types,
            )
            return DiagnosisObservation(
                observation=DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION,
                extra_infos={"performance_id": performance_id},
            )
        logger.info(
            "[PERF-DIAG][master] Performance check finished: "
            "reports_processed=%d, rounds=%d, ready_rounds=%d, "
            "issue_found=False.",
            len(reports),
            len(self._rounds),
            len(self._ready_rounds),
        )
        return None

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> List[DiagnosisAction]:
        if (
            problem is None
            or problem.observation
            != DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
        ):
            logger.info(
                "[PERF-DIAG][master] Resolve skipped: no performance "
                "degradation problem."
            )
            return [NoAction()]

        performance_id = problem.extra_infos.get("performance_id", "")
        performance = self._rounds.get(performance_id)
        if performance is None or performance.state != DiagnosisState.ISSUE:
            logger.info(
                "[PERF-DIAG][master] Resolve skipped: performance_id=%s "
                "is missing or no longer in ISSUE state.",
                performance_id,
            )
            self._discard_ready_round(performance_id)
            return [NoAction()]

        workers = self._active_workers(kwargs)
        if not workers:
            now = int(time.time())
            logger.warning(
                f"No active workers for performance diagnosis "
                f"{performance_id}."
            )
            if (
                now - performance.updated_at
                > self._pending_action_timeout_seconds
            ):
                performance.state = DiagnosisState.CLOSED
                performance.updated_at = now
                performance.explanation = "no active workers for action plan"
                self._discard_ready_round(performance_id)
            return [NoAction()]

        actions = self._build_actions(performance, workers)
        logger.info(
            "[PERF-DIAG][master] Resolving performance issue: "
            "performance_id=%s, workers=%s, actions=%s, "
            "duration_seconds=%s.",
            performance_id,
            [node.id for node in workers],
            [action.action_id for action in actions],
            DiagnosisConstant.PROFILING_DEFAULT_DURATION_SECS,
        )
        performance.action_group_id = performance.performance_id
        performance.planned_action_ids = [
            action.action_id for action in actions
        ]
        # The existing in-memory diagnosis queue receives these actions
        # immediately after resolve returns.
        performance.queued_action_ids = list(performance.planned_action_ids)
        performance.state = DiagnosisState.ACTION
        performance.updated_at = int(time.time())
        performance.explanation = "profiling requested for all active workers"
        performance.state = DiagnosisState.CLOSED
        logger.info(
            f"Request profiling for performance {performance_id} "
            f"on workers {[node.id for node in workers]}."
        )
        self._discard_ready_round(performance_id)
        return actions

    def _process_report(self, report, context):
        if not isinstance(report, PerformanceDiagnosisData):
            logger.info(
                "[PERF-DIAG][master] Ignore unsupported report: "
                "actual_class=%s.",
                report.__class__.__name__,
            )
            return

        now = int(time.time())
        observed_at = self._report_observed_at(report)
        performance = self._find_correlated_round(report, observed_at)
        if performance is None:
            performance = self._new_round(report, observed_at, now)

        self._add_unique(performance.source_report_ids, report.report_id)
        self._add_unique(performance.source_node_ids, report.node_id)
        self._add_unique(performance.issue_types, report.issue_type)
        performance.updated_at = now

        if performance.state in (
            DiagnosisState.NO_ISSUE,
            DiagnosisState.ACTION,
            DiagnosisState.CLOSED,
        ):
            logger.info(
                "[PERF-DIAG][master] Ignore report %s: performance %s "
                "is already in state %s.",
                report.report_id,
                performance.performance_id,
                performance.state,
            )
            return

        invalid_reason = self._validation_error(report, context)
        if invalid_reason:
            logger.info(
                f"Close performance report {report.report_id}: "
                f"{invalid_reason}."
            )
            performance.state = DiagnosisState.NO_ISSUE
            performance.explanation = invalid_reason
            performance.state = DiagnosisState.CLOSED
            self._discard_ready_round(performance.performance_id)
            return

        performance.state = DiagnosisState.ISSUE
        performance.explanation = "training speed degradation confirmed"
        logger.info(
            "[PERF-DIAG][master] Report accepted and correlated: "
            "performance_id=%s, report_id=%s, source_nodes=%s, "
            "source_reports=%s.",
            performance.performance_id,
            report.report_id,
            performance.source_node_ids,
            performance.source_report_ids,
        )
        logger.info(
            f"Confirm performance issue {performance.performance_id} "
            f"from report {report.report_id}."
        )
        if performance.performance_id not in self._ready_rounds:
            self._ready_rounds.append(performance.performance_id)

    def _new_round(self, report, observed_at, now):
        time_bucket = (
            observed_at // self._correlation_window_seconds
            if self._correlation_window_seconds
            else 0
        )
        correlation_key = ":".join(
            [
                self._effective_job_uid(report),
                report.run_id,
                str(report.rdzv_epoch),
                report.config_fingerprint,
                report.issue_type,
                str(time_bucket),
            ]
        )
        performance_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"dlrover-job-performance:{correlation_key}",
            )
        )
        performance = JobPerformance(
            performance_id=performance_id,
            correlation_key=correlation_key,
            state=DiagnosisState.OPEN,
            job_uid=self._effective_job_uid(report),
            run_id=report.run_id,
            rdzv_epoch=report.rdzv_epoch,
            config_fingerprint=report.config_fingerprint,
            observed_at=observed_at,
            created_at=now,
            updated_at=now,
        )
        self._rounds[performance_id] = performance
        self._round_order.append(performance_id)
        while len(self._rounds) > self._max_rounds:
            self._remove_round(self._round_order[0])
        return performance

    def _find_correlated_round(self, report, observed_at):
        for performance_id in reversed(self._round_order):
            performance = self._rounds.get(performance_id)
            if performance is None:
                continue
            if (
                performance.job_uid != self._effective_job_uid(report)
                or performance.run_id != report.run_id
                or performance.rdzv_epoch != report.rdzv_epoch
                or performance.config_fingerprint != report.config_fingerprint
                or report.issue_type not in performance.issue_types
            ):
                continue
            if (
                abs(performance.observed_at - observed_at)
                <= self._correlation_window_seconds
            ):
                return performance
        return None

    def _validation_error(self, report, context):
        if (
            report.issue_type
            != DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
        ):
            return "unsupported performance issue"

        current_job_uid = context.get("job_uid")
        if current_job_uid and current_job_uid != report.job_uid:
            return "report job does not match current job"

        current_epoch = context.get("rdzv_epoch")
        if current_epoch is not None and report.rdzv_epoch != current_epoch:
            return "report rendezvous epoch is stale"

        current_run_id = context.get("run_id")
        if current_run_id and current_run_id != report.run_id:
            return "report run does not match current run"

        if report.sample_count < 1:
            return "report contains no samples"
        for value in (
            report.metric_value,
            report.baseline_value,
            report.threshold_value,
            report.degradation_ratio,
        ):
            if value is not None and not math.isfinite(float(value)):
                return "report contains a non-finite metric"
        if report.baseline_value is not None and report.baseline_value <= 0:
            return "report baseline is not positive"

        workers = (context.get("job_nodes") or {}).get(NodeType.WORKER, {})
        source_node = workers.get(report.node_id)
        if (
            source_node is None
            or source_node.status != NodeStatus.RUNNING
            or source_node.is_released
        ):
            return "report source worker is not active"

        rdzv_node_ids = context.get("rdzv_node_ids")
        if rdzv_node_ids is not None and report.node_id not in set(
            rdzv_node_ids
        ):
            return "report source worker is not in current rendezvous"
        return ""

    def _active_workers(self, context):
        workers = (context.get("job_nodes") or {}).get(NodeType.WORKER, {})
        rdzv_node_ids = context.get("rdzv_node_ids")
        allowed_ids = set(rdzv_node_ids) if rdzv_node_ids is not None else None
        active = []
        for node_id in sorted(workers):
            node = workers[node_id]
            if node.status != NodeStatus.RUNNING or node.is_released:
                continue
            if allowed_ids is not None and node.id not in allowed_ids:
                continue
            active.append(node)
        return active

    def _build_actions(self, performance, workers):
        actions = []
        for node in workers:
            action_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    ":".join(
                        [
                            performance.performance_id,
                            str(node.id),
                            DiagnosisActionType.ENABLE_PROFILING,
                            str(performance.rdzv_epoch),
                        ]
                    ),
                )
            )
            actions.append(
                NodeAction(
                    node_id=node.id,
                    node_type=node.type,
                    node_status=node.status,
                    reason=(DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION),
                    instance=node.id,
                    action_type=DiagnosisActionType.ENABLE_PROFILING,
                    expired_time_period=self._action_expiry_seconds,
                    action_id=action_id,
                    action_group_id=performance.performance_id,
                    diagnosis_id=performance.performance_id,
                    job_uid=performance.job_uid,
                    run_id=performance.run_id,
                    rdzv_epoch=performance.rdzv_epoch,
                    parameters={
                        "duration_seconds": str(
                            DiagnosisConstant.PROFILING_DEFAULT_DURATION_SECS
                        )
                    },
                )
            )
        return actions

    def _expire_pending_rounds(self, now):
        for performance_id in list(self._ready_rounds):
            performance = self._rounds.get(performance_id)
            if performance is None:
                self._discard_ready_round(performance_id)
                continue
            if (
                performance.state == DiagnosisState.ISSUE
                and now - performance.updated_at
                > self._pending_action_timeout_seconds
            ):
                performance.state = DiagnosisState.CLOSED
                performance.updated_at = now
                performance.explanation = "action plan timed out"
                self._discard_ready_round(performance_id)

    def _prune_rounds(self, now):
        while self._round_order:
            performance_id = self._round_order[0]
            performance = self._rounds.get(performance_id)
            if performance is None:
                self._round_order.popleft()
                self._discard_ready_round(performance_id)
                continue
            if (
                performance.state == DiagnosisState.CLOSED
                and now - performance.updated_at
                > self._round_retention_seconds
            ):
                self._remove_round(performance_id)
                continue
            break

        while len(self._rounds) > self._max_rounds:
            self._remove_round(self._round_order[0])

    def _remove_round(self, performance_id):
        if not self._round_order:
            return
        if self._round_order[0] == performance_id:
            self._round_order.popleft()
        else:
            try:
                self._round_order.remove(performance_id)
            except ValueError:
                pass
        self._rounds.pop(performance_id, None)
        self._discard_ready_round(performance_id)

    def _discard_ready_round(self, performance_id):
        self._ready_rounds = deque(
            item for item in self._ready_rounds if item != performance_id
        )

    @staticmethod
    def _add_unique(values, value):
        if value not in values:
            values.append(value)

    @staticmethod
    def _effective_job_uid(report):
        return report.job_uid

    @staticmethod
    def _report_observed_at(report):
        return report.last_timestamp or report.received_at or report.timestamp
