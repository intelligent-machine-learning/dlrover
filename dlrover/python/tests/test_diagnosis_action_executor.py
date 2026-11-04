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

import unittest
from unittest.mock import MagicMock

from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    JobRestartAction,
    NoAction,
    NodeAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.elastic_agent.diagnosis.action_executor import (
    DiagnosisActionExecutor,
)


class DiagnosisActionExecutorTest(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.client = MagicMock()
        self.profiler = MagicMock()
        self.restart = MagicMock()
        self.relaunch = MagicMock()
        self.collect_metric = MagicMock()
        self.executor = DiagnosisActionExecutor(
            agent_context=self.context,
            master_client=self.client,
            restart_worker=self.restart,
            relaunch_worker=self.relaunch,
            collect_metric=self.collect_metric,
            get_diagnosis_context=lambda: {
                "run_id": "run-1",
                "rdzv_epoch": 2,
            },
            node_id=0,
            profiler_controller=self.profiler,
        )

    def _node_action(self, action_type):
        return NodeAction(
            node_id=0,
            node_type="worker",
            instance=0,
            action_type=action_type,
            action_id=f"{action_type}-1",
        )

    def test_poll_dispatches_node_actions(self):
        action = self._node_action(DiagnosisActionType.COLLECT_METRIC)
        self.context.next_diagnosis_action.side_effect = lambda instance: (
            action
            if instance == DiagnosisConstant.LOCAL_INSTANCE
            else NoAction()
        )

        self.executor.poll()
        self.collect_metric.assert_called_once()

        self.executor.execute(
            self._node_action(DiagnosisActionType.RESTART_WORKER)
        )
        self.executor.execute(
            self._node_action(DiagnosisActionType.RELAUNCH_WORKER)
        )
        self.restart.assert_called_once()
        self.relaunch.assert_called_once()

    def test_enable_profiling_uses_action_context(self):
        action = self._node_action(DiagnosisActionType.ENABLE_PROFILING)
        self.profiler.enable.return_value = True

        self.assertTrue(self.executor.execute(action))
        self.profiler.enable.assert_called_once_with(
            action,
            current_run_id="run-1",
            current_rdzv_epoch=2,
        )

    def test_report_actions_use_master_client(self):
        response = MagicMock()
        response.success = True
        self.client.report_diagnosis_data.return_value = response
        report = ReportDiagnosisDataAction(
            data_cls="PerformanceDiagnosisData",
            data_content="{}",
            report_id="report-1",
        )
        self.assertTrue(self.executor.execute(report))
        self.client.report_diagnosis_data.assert_called_once()

        event = EventAction(event_type="type", event_msg="message")
        self.assertTrue(self.executor.execute(event))
        self.client.report_event.assert_called_once()

        job_action = JobRestartAction(reason="test")
        self.assertTrue(self.executor.execute(job_action))
        self.client.report_action.assert_called_once_with(job_action)

        self.client.report_diagnosis_data.side_effect = RuntimeError(
            "master unavailable"
        )
        self.assertFalse(self.executor.execute(report))
        self.context.enqueue_diagnosis_action.assert_called_with(report)

    def test_poll_reenqueues_event_action(self):
        event = EventAction(
            event_type="type",
            event_msg="message",
            expired_time_period=600,
        )
        self.context.next_diagnosis_action.side_effect = lambda instance: (
            event
            if instance == DiagnosisConstant.LOCAL_INSTANCE
            else NoAction()
        )

        self.executor.poll()
        self.client.report_event.assert_called_once()
        self.context.enqueue_diagnosis_action.assert_called_once_with(event)

    def test_poll_does_not_reenqueue_expired_event(self):
        event = EventAction(
            event_type="type",
            event_msg="message",
            timestamp=1,
            expired_time_period=1,
        )
        self.context.next_diagnosis_action.side_effect = lambda instance: (
            event
            if instance == DiagnosisConstant.LOCAL_INSTANCE
            else NoAction()
        )

        self.executor.poll()
        self.client.report_event.assert_called_once()
        self.context.enqueue_diagnosis_action.assert_not_called()

    def test_cleanup_calls_profiler_controller(self):
        self.executor.on_workers_stopping()
        self.executor.shutdown()
        self.profiler.disable.assert_called_once_with(
            reason="workers stopping"
        )
        self.profiler.shutdown.assert_called_once()

    def test_unsupported_and_failed_actions(self):
        self.assertFalse(
            self.executor.execute(None)  # type: ignore[arg-type]
        )
        self.assertFalse(self.executor.execute(DiagnosisAction()))
        self.assertFalse(
            self.executor.execute(self._node_action(DiagnosisActionType.LOG))
        )

        self.profiler.enable.return_value = False
        self.assertFalse(
            self.executor.execute(
                self._node_action(DiagnosisActionType.ENABLE_PROFILING)
            )
        )

    def test_report_rejection_and_missing_acknowledgement(self):
        report = ReportDiagnosisDataAction(
            data_cls="PerformanceDiagnosisData",
            data_content="{}",
            report_id="report-1",
        )
        self.client.report_diagnosis_data.return_value = None
        self.assertFalse(self.executor.execute(report))
        self.context.enqueue_diagnosis_action.assert_called_with(report)

        rejected = MagicMock()
        rejected.success = False
        self.client.report_diagnosis_data.return_value = rejected
        self.assertFalse(self.executor.execute(report))


if __name__ == "__main__":
    unittest.main()
