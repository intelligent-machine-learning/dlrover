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

import os
import unittest
from unittest.mock import patch

from dlrover.python.diagnosis.common.diagnosis_action import (
    NoAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
)
from dlrover.python.diagnosis.diagnostician.performance import (
    ELAPSED_TIME_METRIC,
    THROUGHPUT_METRIC,
    PerformanceHistory,
    PerformanceIssue,
    PerformanceDiagnostician,
    PerformanceDiagnosticianConfig,
    PerformanceRecord,
    PerformanceWindow,
)


class TestPerformanceDiagnostician(unittest.TestCase):
    @staticmethod
    def _log_at(timestamp, iteration, throughput):
        return (
            f"[{timestamp}] "
            "iteration {iteration}/300000 "
            "elapsed time per iteration (ms): 100 | "
            "throughput per GPU (TFLOP/s/GPU): {throughput} | "
            "tokens(B/day/GPU): 1.0"
        ).format(
            iteration=iteration,
            throughput=throughput,
        )

    @classmethod
    def _log(cls, second, iteration, throughput):
        return cls._log_at(
            f"2026-10-15 12:21:{second:02d}",
            iteration,
            throughput,
        )

    @staticmethod
    def _window_logs(start_iteration, throughput, second):
        return [
            TestPerformanceDiagnostician._log(
                second,
                start_iteration + offset,
                throughput,
            )
            for offset in range(3)
        ]

    def _diagnostician(self):
        return PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                window_steps=3,
                history_lookback_seconds=300,
                degradation_fraction_threshold=0.7,
                severe_fraction_threshold=0.1,
                recovery_window_seconds=1,
                recovery_healthy_fraction=1.0,
                minimum_recovery_records=2,
            )
        )

    def test_parse_training_log(self):
        record = PerformanceDiagnostician.parse_log_line(
            self._log(13, 144377, 0.6)
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.iteration, 144377)
        self.assertEqual(record.throughput, 0.6)
        self.assertEqual(record.metrics["tokens_b_per_day_per_gpu"], 1.0)

    def test_requires_degradation_and_deduplicates_tail(self):
        diagnostician = self._diagnostician()
        healthy = self._window_logs(1, 1.0, 1)
        degraded = self._window_logs(4, 0.5, 2)

        self.assertIsNone(diagnostician.observe(logs=healthy))
        observation = diagnostician.observe(logs=degraded)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.observation,
            "training_speed_degradation",
        )

        actions = diagnostician.resolve(
            observation,
            node_id=3,
            node_rank=7,
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=2,
            config_fingerprint="config-1",
        )
        self.assertIsInstance(actions[0], ReportDiagnosisDataAction)
        self.assertEqual(actions[0].node_rank, 7)
        self.assertEqual(actions[0].data_type, "PERFORMANCE_ISSUE")
        self.assertEqual(actions[0].schema_version, 1)
        self.assertTrue(actions[0].report_id)

        data = PerformanceDiagnosisData.from_json(actions[0].data_content)
        self.assertEqual(data.report_id, actions[0].report_id)
        self.assertEqual(data.job_uid, "job-1")
        self.assertEqual(data.run_id, "run-1")
        self.assertEqual(data.rdzv_epoch, 2)
        self.assertEqual(data.node_id, 3)

        serialized_action = ReportDiagnosisDataAction.from_json(
            actions[0].to_json()
        )
        self.assertEqual(serialized_action.report_id, actions[0].report_id)
        self.assertEqual(serialized_action.data_type, actions[0].data_type)
        self.assertIsNone(diagnostician.observe(logs=healthy + degraded))

        repeated_actions = diagnostician.resolve(
            observation,
            node_id=3,
            node_rank=7,
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=2,
            config_fingerprint="config-1",
        )
        self.assertEqual(repeated_actions[0].report_id, actions[0].report_id)
        self.assertEqual(
            repeated_actions[0].data_content,
            actions[0].data_content,
        )

    def test_confirmed_issue_logs_window_details(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(1, 1.0, 1))
        )

        with self.assertLogs("dlrover.logger", level="INFO") as captured:
            observation = diagnostician.observe(
                logs=self._window_logs(4, 0.5, 2)
            )

        self.assertIsNotNone(observation)
        output = "\n".join(captured.output)
        self.assertIn("Performance check started", output)
        self.assertIn("Performance check finished", output)
        self.assertIn("Closed step window", output)
        self.assertIn("steps=4..6", output)
        self.assertIn("previous=1.0", output)
        self.assertIn("rule=sustained_recent", output)
        self.assertIn("Confirmed training speed degradation", output)
        self.assertIn("baseline=1.0", output)
        self.assertIn("threshold=0.7", output)
        self.assertIn("abnormal_fraction=1.000", output)

    def test_checkpoint_iteration_defines_window_origin(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(5001, 1.0, 1))
        )
        windows = diagnostician.window_history
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_iteration, 5001)
        self.assertEqual(windows[0].end_iteration, 5003)

        observation = diagnostician.observe(
            logs=self._window_logs(5004, 0.5, 2)
        )
        self.assertIsNotNone(observation)
        self.assertEqual(
            diagnostician.window_history[-1].start_iteration, 5004
        )

    def test_context_change_resets_checkpoint_windows(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(
                logs=self._window_logs(1, 1.0, 1),
                run_id="run-1",
                rdzv_epoch=1,
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=self._window_logs(5001, 1.0, 2),
                run_id="run-2",
                rdzv_epoch=2,
            )
        )
        self.assertEqual(diagnostician._window_origin_step, 5001)
        self.assertEqual(diagnostician.window_history[0].start_iteration, 5001)

    def test_historical_window_detects_gradual_degradation(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                window_steps=3,
                history_lookback_seconds=5,
                degradation_fraction_threshold=1.0,
                severe_fraction_threshold=1.0,
            )
        )

        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:00", 1, 1.0),
                    self._log_at("2026-10-15 12:00:00", 2, 1.0),
                    self._log_at("2026-10-15 12:00:00", 3, 1.0),
                ]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:03", 4, 0.95),
                    self._log_at("2026-10-15 12:00:03", 5, 0.95),
                    self._log_at("2026-10-15 12:00:03", 6, 0.95),
                ]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:06", 7, 0.9),
                    self._log_at("2026-10-15 12:00:06", 8, 0.9),
                    self._log_at("2026-10-15 12:00:06", 9, 0.9),
                ]
            )
        )

        with self.assertLogs("dlrover.logger", level="INFO") as captured:
            observation = diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:09", 10, 0.65),
                    self._log_at("2026-10-15 12:00:09", 11, 0.65),
                    self._log_at("2026-10-15 12:00:09", 12, 0.65),
                ]
            )

        self.assertIsNotNone(observation)
        output = "\n".join(captured.output)
        self.assertIn("historical=0.95", output)
        self.assertIn("rule=sustained_historical", output)

    def test_consecutive_abnormal_windows_are_required(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                window_steps=3,
                history_lookback_seconds=300,
                degradation_fraction_threshold=1.0,
                severe_fraction_threshold=1.0,
                consecutive_abnormal_windows=2,
            )
        )
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(1, 1.0, 1))
        )
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(4, 0.5, 2))
        )
        observation = diagnostician.observe(logs=self._window_logs(7, 0.5, 3))
        self.assertIsNotNone(observation)

    def test_severe_degradation_uses_q(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                window_steps=3,
                history_lookback_seconds=300,
                degradation_fraction_threshold=0.7,
                severe_fraction_threshold=0.3,
                severe_degradation_ratio=0.2,
            )
        )
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(1, 1.0, 1))
        )
        observation = diagnostician.observe(
            logs=[
                self._log(2, 4, 0.1),
                self._log(2, 5, 1.0),
                self._log(2, 6, 1.0),
            ]
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        issue = PerformanceIssue.from_json(
            observation.extra_infos["performance_issue"]
        )
        self.assertEqual(issue.metric_value, 0.1)
        self.assertAlmostEqual(issue.threshold_value, 0.2)
        self.assertAlmostEqual(issue.degradation_ratio, 0.1)

    def test_recovery_allows_a_new_issue(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(logs=self._window_logs(1, 1.0, 1))
        )
        self.assertIsNotNone(
            diagnostician.observe(logs=self._window_logs(4, 0.5, 2))
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log(3, 7, 1.0),
                    self._log(3, 8, 1.0),
                ]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log(4, 9, 1.0),
                    self._log(4, 10, 1.0),
                ]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[
                    self._log(5, 11, 0.5),
                    self._log(5, 12, 0.5),
                    self._log(5, 13, 0.5),
                ]
            )
        )

    def test_degradation_ratio_is_configurable(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                window_steps=1,
                history_lookback_seconds=300,
                degradation_fraction_threshold=1.0,
                degradation_ratio=0.8,
                severe_fraction_threshold=1.0,
            )
        )
        self.assertIsNone(diagnostician.observe(logs=[self._log(1, 1, 1.0)]))
        self.assertIsNotNone(
            diagnostician.observe(logs=[self._log(2, 2, 0.75)])
        )

    def test_performance_defaults(self):
        config = PerformanceDiagnosticianConfig()
        self.assertEqual(config.window_steps, 100)
        self.assertEqual(config.history_lookback_seconds, 1800)
        self.assertEqual(config.degradation_ratio, 0.7)
        self.assertEqual(config.severe_degradation_ratio, 0.2)
        self.assertEqual(config.consecutive_abnormal_windows, 1)

    def test_config_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "DLROVER_DIAGNOSIS_PERFORMANCE_N": "20",
                "DLROVER_DIAGNOSIS_PERFORMANCE_HISTORY_MINUTES": "1.5",
                "DLROVER_DIAGNOSIS_PERFORMANCE_K": "2",
                "DLROVER_DIAGNOSIS_PERFORMANCE_P": "0.8",
                "DLROVER_DIAGNOSIS_PERFORMANCE_Q": "0.1",
            },
            clear=True,
        ):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config.window_steps, 20)
        self.assertEqual(config.history_lookback_seconds, 90)
        self.assertEqual(config.consecutive_abnormal_windows, 2)
        self.assertEqual(config.degradation_ratio, 0.8)
        self.assertEqual(config.severe_degradation_ratio, 0.1)

    def test_invalid_environment_values_use_defaults(self):
        with patch.dict(
            os.environ,
            {
                "DLROVER_DIAGNOSIS_PERFORMANCE_N": "invalid",
                "DLROVER_DIAGNOSIS_PERFORMANCE_HISTORY_MINUTES": "-1",
                "DLROVER_DIAGNOSIS_PERFORMANCE_K": "0",
                "DLROVER_DIAGNOSIS_PERFORMANCE_P": "1.1",
                "DLROVER_DIAGNOSIS_PERFORMANCE_Q": "invalid",
            },
            clear=True,
        ):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config, PerformanceDiagnosticianConfig())

    def test_config_from_empty_environment_uses_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config, PerformanceDiagnosticianConfig())

    def test_unprefixed_performance_environment_is_ignored(self):
        with patch.dict(
            os.environ,
            {
                "PERFORMANCE_N": "1",
                "PERFORMANCE_K": "9",
                "PERFORMANCE_P": "0.1",
                "PERFORMANCE_Q": "0.1",
            },
            clear=True,
        ):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config, PerformanceDiagnosticianConfig())

    def test_performance_record_and_history(self):
        record = PerformanceRecord(
            timestamp=10,
            iteration=3,
            metrics={THROUGHPUT_METRIC: 2.0},
        )
        self.assertEqual(
            PerformanceRecord.from_json(record.to_json()),
            record,
        )

        history = PerformanceHistory(max_records=2, retention_seconds=10)
        self.assertTrue(history.add(record))
        self.assertFalse(history.add(record))
        metadata_only = PerformanceRecord(timestamp=11, metrics={"x": 1.0})
        self.assertTrue(history.add(metadata_only))
        self.assertEqual(history.throughput_baseline, 2.0)
        self.assertTrue(
            history.add(
                PerformanceRecord(
                    timestamp=12,
                    metrics={THROUGHPUT_METRIC: 1.0},
                )
            )
        )
        self.assertEqual(len(history), 2)
        latest = history.latest
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.timestamp, 12)
        self.assertTrue(
            history.add(
                PerformanceRecord(
                    timestamp=30,
                    metrics={THROUGHPUT_METRIC: 0.5},
                )
            )
        )
        self.assertEqual(len(history), 1)
        history.clear()
        self.assertEqual(len(history), 0)
        self.assertIsNone(history.latest)
        self.assertIsNone(history.throughput_baseline)

    def test_parse_invalid_and_fallback_records(self):
        self.assertIsNone(
            PerformanceDiagnostician.parse_log_line(
                None  # type: ignore[arg-type]
            )
        )
        self.assertIsNone(
            PerformanceDiagnostician.parse_log_line("no metrics here")
        )
        elapsed_only = PerformanceDiagnostician.parse_log_line(
            "elapsed time per iteration (ms): 12.5"
        )
        self.assertIsNotNone(elapsed_only)
        assert elapsed_only is not None
        self.assertEqual(elapsed_only.metrics[ELAPSED_TIME_METRIC], 12.5)
        self.assertIsNone(elapsed_only.iteration)

        with patch(
            "dlrover.python.diagnosis.diagnostician.performance.time.time",
            return_value=123,
        ):
            invalid_time = PerformanceDiagnostician.parse_log_line(
                "[2026-99-99 99:99:99] throughput per GPU (TFLOP/s/GPU): 1.0"
            )
        self.assertIsNotNone(invalid_time)
        assert invalid_time is not None
        self.assertEqual(invalid_time.timestamp, 123)

    def test_collect_training_log_variants(self):
        diagnostician = self._diagnostician()
        training_log = TrainingLog(logs=["line"])
        self.assertIs(
            diagnostician._collect_training_log(training_log=training_log),
            training_log,
        )
        self.assertIsNone(diagnostician._collect_training_log(logs="line"))
        collected_logs = diagnostician._collect_training_log(
            logs=["line-1", "line-2"]
        )
        self.assertIsNotNone(collected_logs)
        assert collected_logs is not None
        self.assertEqual(collected_logs.logs, ["line-1", "line-2"])
        self.assertIsNone(diagnostician._collect_training_log(log_file=1))
        self.assertIsNone(diagnostician._collect_training_log())

        collector = patch(
            "dlrover.python.diagnosis.diagnostician.performance."
            "TrainingLogCollector"
        )
        with collector as collector_cls:
            collector_cls.return_value.collect_data.return_value = training_log
            self.assertIs(
                diagnostician._collect_training_log(log_file="train.log"),
                training_log,
            )

    def test_absolute_thresholds(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                throughput_threshold=2.0,
                severe_throughput_threshold=0.1,
            )
        )
        self.assertEqual(diagnostician._threshold(10.0), 2.0)
        self.assertEqual(diagnostician._severe_threshold(10.0), 0.1)

    def test_empty_and_incomplete_windows(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(window_steps=3)
        )
        self.assertIs(diagnostician.config, diagnostician._config)
        self.assertIs(diagnostician.history, diagnostician._history)
        self.assertEqual(diagnostician.window_history, [])
        self.assertIsNone(diagnostician.observe(logs=[]))
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log(1, 1, 1.0),
                    self._log(1, 2, 1.0),
                ]
            )
        )
        self.assertFalse(diagnostician._is_recovered(None, 10))
        window = PerformanceWindow(
            index=0,
            start_iteration=1,
            end_iteration=3,
            start_timestamp=1,
            end_timestamp=2,
            sample_count=3,
            median_throughput=1.0,
        )
        self.assertIsNone(diagnostician._find_historical_window(window, None))

    def test_resolve_rejects_invalid_observations(self):
        diagnostician = self._diagnostician()
        self.assertIsInstance(
            diagnostician.resolve(None)[0],  # type: ignore[arg-type]
            NoAction,
        )
        self.assertIsInstance(
            diagnostician.resolve(DiagnosisObservation(observation="other"))[
                0
            ],
            NoAction,
        )
        missing = DiagnosisObservation("training_speed_degradation")
        self.assertIsInstance(diagnostician.resolve(missing)[0], NoAction)
        invalid = DiagnosisObservation(
            "training_speed_degradation",
            {"performance_issue": "{"},
        )
        self.assertIsInstance(diagnostician.resolve(invalid)[0], NoAction)


if __name__ == "__main__":
    unittest.main()
