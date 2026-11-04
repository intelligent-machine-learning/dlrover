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

import json
import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    EnvConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)


THROUGHPUT_METRIC = "throughput_per_gpu_tflops"
ELAPSED_TIME_METRIC = "elapsed_time_per_iteration_ms"

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TIMESTAMP_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"
)
_ITERATION_PATTERN = re.compile(r"\biteration\s*:?\s*(\d+)", re.IGNORECASE)
_THROUGHPUT_PATTERN = re.compile(
    rf"throughput\s+per\s+GPU\s*\(TFLOP/s/GPU\)\s*:\s*({_NUMBER})",
    re.IGNORECASE,
)
_ELAPSED_TIME_PATTERN = re.compile(
    rf"elapsed\s+time\s+per\s+iteration\s*\(ms\)\s*:\s*({_NUMBER})",
    re.IGNORECASE,
)


def _normalise_metric_name(name: str) -> str:
    name = name.strip().lower().replace("/", "_per_")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _parse_log_timestamp(log_line: str) -> int:
    matches = _TIMESTAMP_PATTERN.findall(log_line)
    if not matches:
        return int(time.time())
    try:
        return int(
            datetime.fromisoformat(matches[-1].replace(" ", "T")).timestamp()
        )
    except ValueError:
        return int(time.time())


def _get_int_env(env_key: str, default: int, minimum: int) -> int:
    raw_value = env_utils.get_env(env_key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = minimum - 1
    if value < minimum:
        logger.warning(
            f"Invalid {env_key}={raw_value}; use default {default}."
        )
        return default
    return value


def _get_float_env(
    env_key: str,
    default: float,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    raw_value = env_utils.get_env(env_key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")
    if (
        not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        logger.warning(
            f"Invalid {env_key}={raw_value}; use default {default}."
        )
        return default
    return value


@dataclass(frozen=True)
class PerformanceRecord:
    """One performance observation.

    ``iteration`` is optional because some performance signals are emitted
    outside an iteration.  ``metrics`` keeps the record extensible while the
    first implementation focuses on ``THROUGHPUT_METRIC``.
    """

    timestamp: int
    iteration: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def throughput(self) -> Optional[float]:
        return self.metrics.get(THROUGHPUT_METRIC)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PerformanceRecord":
        iteration = value.get("iteration")
        return cls(
            timestamp=int(value.get("timestamp", 0)),
            iteration=int(iteration) if iteration is not None else None,
            metrics={
                str(name): float(metric_value)
                for name, metric_value in value.get("metrics", {}).items()
            },
        )

    @classmethod
    def from_json(cls, value: str) -> "PerformanceRecord":
        return cls.from_dict(json.loads(value))

    @property
    def signature(self) -> Tuple[object, ...]:
        """Stable identity used to deduplicate repeated log tails."""
        return (
            self.timestamp,
            self.iteration,
            tuple(sorted(self.metrics.items())),
        )


class PerformanceHistory:
    """Bounded recent history with duplicate and age eviction."""

    def __init__(self, max_records: int = 1024, retention_seconds: int = 1800):
        self._max_records = max(1, max_records)
        self._retention_seconds = max(0, retention_seconds)
        self._records: Deque[PerformanceRecord] = deque(
            maxlen=self._max_records
        )
        self._record_keys: Set[Tuple[object, ...]] = set()

    @property
    def records(self) -> List[PerformanceRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: PerformanceRecord) -> bool:
        """Add ``record`` and return ``False`` when it is a duplicate."""
        if record.signature in self._record_keys:
            return False

        while self._records and (
            record.timestamp - self._records[0].timestamp
            > self._retention_seconds
        ):
            old_record = self._records.popleft()
            self._record_keys.discard(old_record.signature)

        if len(self._records) == self._max_records:
            old_record = self._records.popleft()
            self._record_keys.discard(old_record.signature)

        self._records.append(record)
        self._record_keys.add(record.signature)
        return True

    def clear(self):
        self._records.clear()
        self._record_keys.clear()

    @property
    def latest(self) -> Optional[PerformanceRecord]:
        return self._records[-1] if self._records else None

    @property
    def throughput_baseline(self) -> Optional[float]:
        values: List[float] = []
        for record in self._records:
            throughput = record.throughput
            if throughput is not None:
                values.append(throughput)
        return median(values) if values else None


@dataclass(frozen=True)
class PerformanceWindow:
    """Summary of one complete, non-overlapping step window."""

    index: int
    start_iteration: int
    end_iteration: int
    start_timestamp: int
    end_timestamp: int
    sample_count: int
    median_throughput: float


@dataclass
class PerformanceIssue:
    """A performance problem supported by a sustained set of observations."""

    issue_type: str = DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
    metric_name: str = THROUGHPUT_METRIC
    metric_value: Optional[float] = None
    baseline_value: Optional[float] = None
    threshold_value: Optional[float] = None
    degradation_ratio: Optional[float] = None
    sample_count: int = 0
    first_timestamp: int = 0
    last_timestamp: int = 0
    first_iteration: Optional[int] = None
    last_iteration: Optional[int] = None
    evidence: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "issue_type": self.issue_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "threshold_value": self.threshold_value,
            "degradation_ratio": self.degradation_ratio,
            "sample_count": self.sample_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "first_iteration": self.first_iteration,
            "last_iteration": self.last_iteration,
            "evidence": self.evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PerformanceIssue":
        return cls(**value)

    @classmethod
    def from_json(cls, value: str) -> "PerformanceIssue":
        return cls.from_dict(json.loads(value))


@dataclass
class PerformanceDiagnosticianConfig:
    """Configuration shared by the worker-side performance diagnostician."""

    log_tail_lines: int = 5000
    history_size: int = 1024
    history_retention_seconds: int = 3600

    # p in the performance rule. If an absolute threshold is not configured,
    # each reference threshold is reference_throughput * degradation_ratio.
    throughput_threshold: Optional[float] = None
    degradation_ratio: float = 0.7

    # n in the performance rule: number of steps in each fixed window.
    window_steps: int = 100
    history_lookback_seconds: int = 30 * 60

    # Fraction of samples in a window that must be below the degradation
    # threshold for the window to be abnormal.
    degradation_fraction_threshold: float = 0.70

    # k in both performance rules. Only consecutive, non-overlapping
    # abnormal windows are counted.
    consecutive_abnormal_windows: int = 1

    # q in the severe rule. The severe threshold is
    # reference_throughput * severe_degradation_ratio unless an absolute
    # threshold is configured. Severe windows reuse n and k.
    severe_throughput_threshold: Optional[float] = None
    severe_degradation_ratio: float = 0.20
    severe_fraction_threshold: float = 0.10

    # Recovery is also window based to avoid clearing an issue after
    # only a few lucky healthy samples.
    recovery_window_seconds: int = 60
    recovery_healthy_fraction: float = 0.90
    minimum_recovery_records: int = 10

    maximum_evidence_records: int = 5

    @classmethod
    def from_env(cls) -> "PerformanceDiagnosticianConfig":
        """Build a config from the prefixed performance environment."""
        config = cls()
        config.window_steps = _get_int_env(
            EnvConfigKey.DLROVER_DIAGNOSIS_PERFORMANCE_N,
            config.window_steps,
            1,
        )
        default_history_minutes = config.history_lookback_seconds / 60.0
        history_minutes = _get_float_env(
            EnvConfigKey.DLROVER_DIAGNOSIS_PERFORMANCE_HISTORY_MINUTES,
            default_history_minutes,
            0.0,
        )
        config.history_lookback_seconds = int(round(history_minutes * 60))
        config.consecutive_abnormal_windows = _get_int_env(
            EnvConfigKey.DLROVER_DIAGNOSIS_PERFORMANCE_K,
            config.consecutive_abnormal_windows,
            1,
        )
        config.degradation_ratio = _get_float_env(
            EnvConfigKey.DLROVER_DIAGNOSIS_PERFORMANCE_P,
            config.degradation_ratio,
            0.0,
            1.0,
        )
        config.severe_degradation_ratio = _get_float_env(
            EnvConfigKey.DLROVER_DIAGNOSIS_PERFORMANCE_Q,
            config.severe_degradation_ratio,
            0.0,
            1.0,
        )
        return config


class PerformanceDiagnostician(Diagnostician):
    """Diagnose training-speed degradation on one worker.

    Detection uses two independent rules:

    1. sustained degradation: a high fraction of samples in fixed step
       windows are below p times a reference window;
    2. severe intermittent degradation: a smaller fraction of samples in the
       same windows are below q times a reference window.

    Each window is compared with the previous window and a configurable
    historical window. Both rules confirm an issue after k consecutive
    abnormal windows.

    The existing TrainingLogCollector, PerformanceHistory, report path, and
    PerformanceIssue/PerformanceDiagnosisData structures are reused.
    """

    def __init__(
        self, config: Optional[PerformanceDiagnosticianConfig] = None
    ):
        super().__init__()
        self._config = config or PerformanceDiagnosticianConfig()
        self._history = PerformanceHistory(
            self._config.history_size,
            self._config.history_retention_seconds,
        )
        self._issue_reported = False
        self._recovery_baseline: Optional[float] = None
        self._window_origin_step: Optional[int] = None
        self._window_index = 0
        self._window_start_timestamp: Optional[int] = None
        self._window_records: List[PerformanceRecord] = []
        self._window_history: Deque[PerformanceWindow] = deque()
        self._consecutive_sustained_windows = 0
        self._consecutive_severe_windows = 0
        self._sustained_sequence_reference: Optional[PerformanceWindow] = None
        self._severe_sequence_reference: Optional[PerformanceWindow] = None
        self._last_iteration: Optional[int] = None
        self._context_key: Optional[Tuple[str, str, int, str]] = None

    @property
    def config(self) -> PerformanceDiagnosticianConfig:
        return self._config

    @property
    def history(self) -> PerformanceHistory:
        return self._history

    @property
    def window_history(self) -> List[PerformanceWindow]:
        return list(self._window_history)

    @staticmethod
    def parse_log_line(log_line: str) -> Optional[PerformanceRecord]:
        """Parse one training log line into a generic performance record."""
        if not isinstance(log_line, str):
            return None

        metrics: Dict[str, float] = {}
        throughput_match = _THROUGHPUT_PATTERN.search(log_line)
        if throughput_match:
            metrics[THROUGHPUT_METRIC] = float(throughput_match.group(1))

        elapsed_match = _ELAPSED_TIME_PATTERN.search(log_line)
        if elapsed_match:
            metrics[ELAPSED_TIME_METRIC] = float(elapsed_match.group(1))

        # Additional numeric fields after the first pipe are retained for
        # future performance checks.  The first segment contains the log
        # prefix and iteration number, so it is deliberately excluded.
        for segment in log_line.split("|")[1:]:
            match = re.search(rf"([^:|]+):\s*({_NUMBER})\s*$", segment.strip())
            if not match:
                continue
            metric_name = _normalise_metric_name(match.group(1))
            if metric_name:
                metrics.setdefault(metric_name, float(match.group(2)))

        if not metrics:
            return None

        iteration_match = _ITERATION_PATTERN.search(log_line)
        iteration = int(iteration_match.group(1)) if iteration_match else None
        return PerformanceRecord(
            timestamp=_parse_log_timestamp(log_line),
            iteration=iteration,
            metrics=metrics,
        )

    def _collect_training_log(self, **kwargs) -> Optional[TrainingLog]:
        training_log = kwargs.get("training_log")
        if isinstance(training_log, TrainingLog):
            return training_log

        logs = kwargs.get("logs")
        if logs is not None:
            if isinstance(logs, (str, bytes)) or not isinstance(
                logs, Iterable
            ):
                return None
            return TrainingLog(logs=[str(log) for log in logs])

        log_file = kwargs.get("log_file")
        if not log_file:
            return None
        if not isinstance(log_file, str):
            return None
        return TrainingLogCollector(
            log_file=log_file,
            n_line=self._config.log_tail_lines,
        ).collect_data()

    def _threshold(self, baseline: Optional[float]) -> Optional[float]:
        if self._config.throughput_threshold is not None:
            return self._config.throughput_threshold
        if baseline is None:
            return None
        return baseline * self._config.degradation_ratio

    def _severe_threshold(self, baseline: Optional[float]) -> Optional[float]:
        if self._config.severe_throughput_threshold is not None:
            return self._config.severe_throughput_threshold
        if baseline is None:
            return None
        return baseline * self._config.severe_degradation_ratio

    def _throughput_records_in_window(
        self, now: int, window_seconds: int
    ) -> List[PerformanceRecord]:
        start = now - max(0, window_seconds)
        return [
            item
            for item in self._history.records
            if item.throughput is not None and start <= item.timestamp <= now
        ]

    def _reset_performance_windows(self):
        self._issue_reported = False
        self._recovery_baseline = None
        self._window_origin_step = None
        self._window_index = 0
        self._window_start_timestamp = None
        self._window_records.clear()
        self._window_history.clear()
        self._consecutive_sustained_windows = 0
        self._consecutive_severe_windows = 0
        self._sustained_sequence_reference = None
        self._severe_sequence_reference = None
        self._last_iteration = None
        self._history.clear()

    def _update_context(self, **kwargs):
        context_key = (
            str(kwargs.get("job_uid", "") or ""),
            str(kwargs.get("run_id", "") or ""),
            int(kwargs.get("rdzv_epoch", 0) or 0),
            str(kwargs.get("config_fingerprint", "") or ""),
        )
        if self._context_key is None:
            self._context_key = context_key
            return
        if context_key != self._context_key:
            logger.info(
                "[PERF-DIAG][worker] Reset performance windows because "
                "the execution context changed: old=%s, new=%s.",
                self._context_key,
                context_key,
            )
            self._reset_performance_windows()
            self._context_key = context_key

    def _find_historical_window(
        self,
        current: PerformanceWindow,
        previous: Optional[PerformanceWindow],
    ) -> Optional[PerformanceWindow]:
        cutoff = current.end_timestamp - self._config.history_lookback_seconds
        for window in reversed(self._window_history):
            if window.end_timestamp <= cutoff:
                if previous is not None and window.index == previous.index:
                    return None
                return window
        return None

    def _append_window(self, window: PerformanceWindow):
        self._window_history.append(window)
        retention_seconds = max(
            self._config.history_retention_seconds,
            self._config.history_lookback_seconds * 2,
        )
        cutoff = window.end_timestamp - retention_seconds
        while (
            len(self._window_history) > 1
            and self._window_history[0].end_timestamp < cutoff
        ):
            self._window_history.popleft()

    def _compare_window(
        self,
        records: List[PerformanceRecord],
        reference: PerformanceWindow,
        ratio: float,
        absolute_threshold: Optional[float],
        fraction_threshold: float,
    ):
        baseline = reference.median_throughput
        threshold = (
            absolute_threshold
            if absolute_threshold is not None
            else baseline * ratio
        )
        abnormal = [
            record
            for record in records
            if record.throughput is not None and record.throughput < threshold
        ]
        fraction = len(abnormal) / max(1, len(records))
        return fraction >= fraction_threshold, fraction, threshold, abnormal

    def _close_current_window(self) -> Optional[PerformanceIssue]:
        if self._window_origin_step is None or not self._window_records:
            return None

        throughputs = [
            record.throughput
            for record in self._window_records
            if record.throughput is not None
        ]
        if not throughputs:
            self._window_records.clear()
            self._window_index += 1
            return None

        start_iteration = (
            self._window_origin_step
            + self._window_index * self._config.window_steps
        )
        end_iteration = start_iteration + self._config.window_steps - 1
        current = PerformanceWindow(
            index=self._window_index,
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            start_timestamp=self._window_start_timestamp
            or self._window_records[0].timestamp,
            end_timestamp=self._window_records[-1].timestamp,
            sample_count=len(throughputs),
            median_throughput=median(throughputs),
        )
        previous = self._window_history[-1] if self._window_history else None
        historical = self._find_historical_window(current, previous)
        records = list(self._window_records)

        recent_sustained = None
        historical_sustained = None
        recent_severe = None
        historical_severe = None
        sustained_sequence = None
        severe_sequence = None
        if previous is not None:
            recent_sustained = self._compare_window(
                records,
                previous,
                self._config.degradation_ratio,
                self._config.throughput_threshold,
                self._config.degradation_fraction_threshold,
            )
            recent_severe = self._compare_window(
                records,
                previous,
                self._config.severe_degradation_ratio,
                self._config.severe_throughput_threshold,
                self._config.severe_fraction_threshold,
            )
        if historical is not None:
            historical_sustained = self._compare_window(
                records,
                historical,
                self._config.degradation_ratio,
                self._config.throughput_threshold,
                self._config.degradation_fraction_threshold,
            )
            historical_severe = self._compare_window(
                records,
                historical,
                self._config.severe_degradation_ratio,
                self._config.severe_throughput_threshold,
                self._config.severe_fraction_threshold,
            )
        if (
            self._consecutive_sustained_windows > 0
            and self._sustained_sequence_reference is not None
        ):
            sustained_sequence = self._compare_window(
                records,
                self._sustained_sequence_reference,
                self._config.degradation_ratio,
                self._config.throughput_threshold,
                self._config.degradation_fraction_threshold,
            )
        if (
            self._consecutive_severe_windows > 0
            and self._severe_sequence_reference is not None
        ):
            severe_sequence = self._compare_window(
                records,
                self._severe_sequence_reference,
                self._config.severe_degradation_ratio,
                self._config.severe_throughput_threshold,
                self._config.severe_fraction_threshold,
            )

        sustained_candidates = []
        severe_candidates = []
        for source, result in (
            ("recent", recent_sustained),
            ("historical", historical_sustained),
            ("sequence", sustained_sequence),
        ):
            if result is not None and result[0]:
                reference = (
                    previous
                    if source == "recent"
                    else historical
                    if source == "historical"
                    else self._sustained_sequence_reference
                )
                sustained_candidates.append(
                    (result[1], source, result, reference)
                )
        for source, result in (
            ("recent", recent_severe),
            ("historical", historical_severe),
            ("sequence", severe_sequence),
        ):
            if result is not None and result[0]:
                reference = (
                    previous
                    if source == "recent"
                    else historical
                    if source == "historical"
                    else self._severe_sequence_reference
                )
                severe_candidates.append(
                    (result[1], source, result, reference)
                )

        sustained_any = bool(sustained_candidates)
        severe_any = bool(severe_candidates)
        if sustained_any:
            if self._consecutive_sustained_windows == 0:
                _, _, _, reference = max(
                    sustained_candidates, key=lambda item: item[0]
                )
                self._sustained_sequence_reference = reference
            self._consecutive_sustained_windows += 1
        else:
            self._consecutive_sustained_windows = 0
            self._sustained_sequence_reference = None
        if severe_any:
            if self._consecutive_severe_windows == 0:
                _, _, _, reference = max(
                    severe_candidates, key=lambda item: item[0]
                )
                self._severe_sequence_reference = reference
            self._consecutive_severe_windows += 1
        else:
            self._consecutive_severe_windows = 0
            self._severe_sequence_reference = None

        logger.info(
            "[PERF-DIAG][worker] Closed step window: "
            "window=%d, steps=%d..%d, samples=%d, median=%s, "
            "previous=%s, historical=%s, "
            "recent_sustained_fraction=%s, "
            "historical_sustained_fraction=%s, "
            "sequence_sustained_fraction=%s, "
            "recent_severe_fraction=%s, "
            "historical_severe_fraction=%s, "
            "sequence_severe_fraction=%s, "
            "sustained_consecutive=%d, severe_consecutive=%d.",
            current.index,
            current.start_iteration,
            current.end_iteration,
            current.sample_count,
            current.median_throughput,
            previous.median_throughput if previous else None,
            historical.median_throughput if historical else None,
            recent_sustained[1] if recent_sustained else None,
            historical_sustained[1] if historical_sustained else None,
            sustained_sequence[1] if sustained_sequence else None,
            recent_severe[1] if recent_severe else None,
            historical_severe[1] if historical_severe else None,
            severe_sequence[1] if severe_sequence else None,
            self._consecutive_sustained_windows,
            self._consecutive_severe_windows,
        )

        selected = None
        selected_kind = ""
        selected_source = ""
        if severe_candidates and self._consecutive_severe_windows >= max(
            1, self._config.consecutive_abnormal_windows
        ):
            _, selected_source, result, reference = max(
                severe_candidates, key=lambda item: item[0]
            )
            selected = result
            selected_kind = "severe"
        elif (
            sustained_candidates
            and self._consecutive_sustained_windows
            >= max(1, self._config.consecutive_abnormal_windows)
        ):
            _, selected_source, result, reference = max(
                sustained_candidates, key=lambda item: item[0]
            )
            selected = result
            selected_kind = "sustained"

        issue = None
        if selected is not None:
            _, fraction, threshold, abnormal_records = selected
            issue = self._make_issue(
                records[-1],
                reference.median_throughput if reference else None,
                threshold,
                abnormal_records,
            )
            self._issue_reported = True
            self._recovery_baseline = (
                reference.median_throughput if reference else None
            )
            self._consecutive_sustained_windows = 0
            self._consecutive_severe_windows = 0
            self._sustained_sequence_reference = None
            self._severe_sequence_reference = None
            self._history.clear()
            rule = f"{selected_kind}_{selected_source}"
            self._log_confirmed_issue(
                rule,
                issue,
                fraction,
                records[-1],
            )

        self._append_window(current)
        self._window_records.clear()
        self._window_index += 1
        self._window_start_timestamp = None
        return issue

    # Recovery uses a recent window instead of N consecutive samples.
    def _is_recovered(
        self,
        baseline: Optional[float],
        now: int,
    ) -> bool:
        records = self._throughput_records_in_window(
            now, self._config.recovery_window_seconds
        )
        if len(records) < max(1, self._config.minimum_recovery_records):
            return False

        threshold = self._threshold(baseline)
        if threshold is None:
            return False

        healthy_count = sum(
            1
            for item in records
            if item.throughput is not None and item.throughput >= threshold
        )
        healthy_fraction = healthy_count / len(records)
        return healthy_fraction >= self._config.recovery_healthy_fraction

    # Build evidence from the records that actually triggered the
    # selected rule. The metric value is the worst observed throughput rather
    # than necessarily the most recent sample.
    def _make_issue(
        self,
        record: PerformanceRecord,
        baseline: Optional[float],
        threshold: Optional[float],
        abnormal_records: List[PerformanceRecord],
        sample_count: Optional[int] = None,
    ) -> PerformanceIssue:
        evidence_records = abnormal_records or [record]
        evidence_records = sorted(
            evidence_records, key=lambda item: item.timestamp
        )

        representative = min(
            evidence_records,
            key=lambda item: (
                item.throughput
                if item.throughput is not None
                else float("inf")
            ),
        )
        throughput = representative.throughput
        ratio = (
            throughput / baseline
            if throughput is not None
            and baseline is not None
            and baseline != 0
            else None
        )
        evidence = [
            item.to_dict()
            for item in evidence_records[
                -self._config.maximum_evidence_records :
            ]
        ]

        return PerformanceIssue(
            metric_value=representative.throughput,
            baseline_value=baseline,
            threshold_value=threshold,
            degradation_ratio=ratio,
            sample_count=(
                len(abnormal_records) if sample_count is None else sample_count
            ),
            first_timestamp=evidence_records[0].timestamp,
            last_timestamp=evidence_records[-1].timestamp,
            first_iteration=evidence_records[0].iteration,
            last_iteration=evidence_records[-1].iteration,
            evidence=evidence,
        )

    def _log_confirmed_issue(
        self,
        rule: str,
        issue: PerformanceIssue,
        abnormal_fraction: float,
        trigger_record: PerformanceRecord,
    ):
        logger.info(
            "Confirmed training speed degradation on worker %s: "
            "rule=%s, metric=%s, trigger_throughput=%s, "
            "worst_throughput=%s, baseline=%s, threshold=%s, "
            "degradation_ratio=%s, abnormal_fraction=%.3f, "
            "abnormal_samples=%s, iterations=%s..%s, "
            "timestamps=%s..%s.",
            env_utils.get_node_id(),
            rule,
            issue.metric_name,
            trigger_record.throughput,
            issue.metric_value,
            issue.baseline_value,
            issue.threshold_value,
            issue.degradation_ratio,
            abnormal_fraction,
            issue.sample_count,
            issue.first_iteration,
            issue.last_iteration,
            issue.first_timestamp,
            issue.last_timestamp,
        )

    def _advance_record(
        self, record: PerformanceRecord
    ) -> Optional[PerformanceIssue]:
        iteration = record.iteration
        if iteration is None or record.throughput is None:
            return None

        if self._window_origin_step is None:
            self._window_origin_step = iteration
            self._window_start_timestamp = record.timestamp

        expected_start = (
            self._window_origin_step
            + self._window_index * self._config.window_steps
        )
        expected_end = expected_start + self._config.window_steps - 1
        if iteration < expected_start:
            return None

        if iteration > expected_end:
            issue = self._close_current_window()
            if issue is not None:
                return issue
            self._window_index = (
                iteration - self._window_origin_step
            ) // self._config.window_steps
            self._window_start_timestamp = record.timestamp
            expected_end = (
                self._window_origin_step
                + (self._window_index + 1) * self._config.window_steps
                - 1
            )

        if not self._window_records:
            self._window_start_timestamp = record.timestamp
        self._window_records.append(record)
        if iteration >= expected_end:
            return self._close_current_window()
        return None

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        self._update_context(**kwargs)
        log_file = kwargs.get("log_file", "")
        logger.info(
            "[PERF-DIAG][worker] Performance check started: "
            "node=%s, log_file=%s, history_records=%d, "
            "issue_reported=%s, window_steps=%d, "
            "history_lookback_seconds=%d, origin_step=%s, "
            "window_index=%d, current_window_records=%d, "
            "k=%d, p=%s, q=%s.",
            env_utils.get_node_id(),
            log_file,
            len(self._history),
            self._issue_reported,
            self._config.window_steps,
            self._config.history_lookback_seconds,
            self._window_origin_step,
            self._window_index,
            len(self._window_records),
            self._config.consecutive_abnormal_windows,
            self._config.degradation_ratio,
            self._config.severe_degradation_ratio,
        )
        training_log = self._collect_training_log(**kwargs)
        if training_log is None or not training_log.logs:
            logger.info(
                "[PERF-DIAG][worker] Performance check skipped: "
                "no training log lines were collected."
            )
            return None

        issue = None
        parsed_records = 0
        duplicate_records = 0
        new_records = 0
        invalid_records = 0
        last_record = None
        for log_line in training_log.logs:
            record = self.parse_log_line(log_line)
            if record is None:
                continue
            parsed_records += 1
            if record.iteration is None or record.throughput is None:
                invalid_records += 1
                continue
            if (
                self._last_iteration is not None
                and record.iteration <= self._last_iteration
            ):
                duplicate_records += 1
                continue
            if not self._history.add(record):
                duplicate_records += 1
                continue
            self._last_iteration = record.iteration
            new_records += 1
            last_record = record
            if self._issue_reported and not self._is_recovered(
                self._recovery_baseline,
                record.timestamp,
            ):
                continue
            if self._issue_reported:
                logger.info(
                    "[PERF-DIAG][worker] Performance issue recovered at "
                    "iteration=%s; reset performance windows.",
                    record.iteration,
                )
                self._reset_performance_windows()
                self._history.add(record)
                self._last_iteration = record.iteration
            candidate = self._advance_record(record)
            logger.debug(
                "[PERF-DIAG][worker] Evaluated record: "
                "timestamp=%s, iteration=%s, throughput=%s, "
                "origin_step=%s, window_index=%d, "
                "current_window_records=%d, "
                "sustained_consecutive=%d, "
                "severe_consecutive=%d, issue=%s.",
                record.timestamp,
                record.iteration,
                record.throughput,
                self._window_origin_step,
                self._window_index,
                len(self._window_records),
                self._consecutive_sustained_windows,
                self._consecutive_severe_windows,
                candidate is not None,
            )
            if issue is None and candidate is not None:
                issue = candidate

        logger.info(
            "[PERF-DIAG][worker] Performance check finished: "
            "collected_lines=%d, parsed_records=%d, "
            "invalid_records=%d, duplicate_records=%d, "
            "new_records=%d, "
            "last_timestamp=%s, last_iteration=%s, "
            "last_throughput=%s, origin_step=%s, "
            "window_index=%d, current_window_records=%d, "
            "sustained_consecutive=%d, severe_consecutive=%d, "
            "issue_found=%s.",
            len(training_log.logs),
            parsed_records,
            invalid_records,
            duplicate_records,
            new_records,
            last_record.timestamp if last_record else None,
            last_record.iteration if last_record else None,
            last_record.throughput if last_record else None,
            self._window_origin_step,
            self._window_index,
            len(self._window_records),
            self._consecutive_sustained_windows,
            self._consecutive_severe_windows,
            issue is not None,
        )
        if issue is None:
            return None
        return DiagnosisObservation(
            observation=issue.issue_type,
            extra_infos={"performance_issue": issue.to_json()},
        )

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> List[DiagnosisAction]:
        if (
            problem is None
            or problem.observation
            != DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
        ):
            return [NoAction()]

        issue_json = problem.extra_infos.get("performance_issue")
        if not issue_json:
            return [NoAction()]
        try:
            issue = PerformanceIssue.from_json(issue_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [NoAction()]

        job_uid = str(kwargs.get("job_uid", env_utils.get_job_uid()) or "")
        run_id = str(
            kwargs.get("run_id")
            or env_utils.get_env("TORCHELASTIC_RUN_ID")
            or ""
        )
        rdzv_epoch = int(kwargs.get("rdzv_epoch", 0) or 0)
        config_fingerprint = str(kwargs.get("config_fingerprint", "") or "")
        node_id = int(kwargs.get("node_id", env_utils.get_node_id()))
        node_rank = int(kwargs.get("node_rank", env_utils.get_node_rank()))
        report_key = ":".join(
            [
                job_uid,
                run_id,
                str(rdzv_epoch),
                config_fingerprint,
                str(node_id),
                str(node_rank),
                issue.issue_type,
                str(issue.first_timestamp),
                str(issue.last_iteration),
            ]
        )
        report_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"dlrover-performance:{report_key}")
        )

        data = PerformanceDiagnosisData(
            timestamp=issue.last_timestamp or int(time.time()),
            issue_type=issue.issue_type,
            metric_name=issue.metric_name,
            metric_value=issue.metric_value,
            baseline_value=issue.baseline_value,
            threshold_value=issue.threshold_value,
            degradation_ratio=issue.degradation_ratio,
            sample_count=issue.sample_count,
            first_timestamp=issue.first_timestamp,
            last_timestamp=issue.last_timestamp,
            first_iteration=issue.first_iteration,
            last_iteration=issue.last_iteration,
            evidence=issue.evidence,
            node_id=node_id,
            node_rank=node_rank,
            report_id=report_id,
            job_uid=job_uid,
            run_id=run_id,
            rdzv_epoch=rdzv_epoch,
            config_fingerprint=config_fingerprint,
        )
        logger.info(
            "[PERF-DIAG][worker] Resolved performance issue into report: "
            "node=%s, rank=%s, report_id=%s, metric=%s, "
            "value=%s, baseline=%s, threshold=%s, ratio=%s, "
            "abnormal_samples=%s, iterations=%s..%s.",
            node_id,
            node_rank,
            report_id,
            issue.metric_name,
            issue.metric_value,
            issue.baseline_value,
            issue.threshold_value,
            issue.degradation_ratio,
            issue.sample_count,
            issue.first_iteration,
            issue.last_iteration,
        )
        return [
            ReportDiagnosisDataAction(
                data_cls=data.__class__.__name__,
                data_content=data.to_json(),
                node_rank=data.node_rank,
                reason=issue.issue_type,
                report_id=data.report_id,
                data_type=data.data_type,
                schema_version=data.schema_version,
            )
        ]
