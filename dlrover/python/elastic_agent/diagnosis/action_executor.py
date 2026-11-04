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

from datetime import datetime
from typing import Callable, Dict, Optional

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    JobAction,
    NoAction,
    NodeAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.elastic_agent.profiler_controller import (
    ProfilerController,
)
from dlrover.python.util.time_util import timestamp_diff_in_seconds


class DiagnosisActionExecutor:
    """Execute diagnosis actions received by one worker agent."""

    def __init__(
        self,
        agent_context,
        master_client,
        restart_worker: Callable[[NodeAction], None],
        relaunch_worker: Callable[[NodeAction], None],
        collect_metric: Callable[[], None],
        get_diagnosis_context: Optional[Callable[[], Dict]] = None,
        node_id=None,
        profiler_controller=None,
    ):
        self._agent_context = agent_context
        self._master_client = master_client
        self._restart_worker = restart_worker
        self._relaunch_worker = relaunch_worker
        self._collect_metric = collect_metric
        self._get_diagnosis_context = get_diagnosis_context or (lambda: {})
        self._node_id = env_utils.get_node_id() if node_id is None else node_id
        self._profiler_controller = profiler_controller or ProfilerController(
            node_id=self._node_id
        )

    def poll(self):
        """Process at most one action for each executable instance."""
        for instance in (
            DiagnosisConstant.LOCAL_INSTANCE,
            DiagnosisConstant.ANY_WORKER_INSTANCE,
            self._node_id,
        ):
            action = self._agent_context.next_diagnosis_action(instance)
            if isinstance(action, NoAction):
                continue
            logger.info(f"Start processing diagnosis action: {action}")
            self.execute(action)
            if isinstance(action, EventAction) and not action.is_expired():
                self._reenqueue_event(action)

    def execute(self, action: DiagnosisAction):
        if not action or isinstance(action, NoAction):
            return False
        if isinstance(action, NodeAction):
            return self._execute_node_action(action)
        if isinstance(action, ReportDiagnosisDataAction):
            return self._report_diagnosis_data(action)
        if isinstance(action, EventAction):
            return self._report_event(action)
        if isinstance(action, JobAction):
            return self._report_job_action(action)

        logger.warning(
            f"Unsupported diagnosis action: {action.__class__.__name__}"
        )
        return False

    def on_workers_stopping(self):
        self._profiler_controller.disable(reason="workers stopping")

    def shutdown(self):
        self._profiler_controller.shutdown()

    def _execute_node_action(self, action: NodeAction):
        action_type = action.action_type
        if action_type == DiagnosisActionType.RESTART_WORKER:
            self._restart_worker(action)
            return True
        if action_type == DiagnosisActionType.RELAUNCH_WORKER:
            self._relaunch_worker(action)
            return True
        if action_type == DiagnosisActionType.COLLECT_METRIC:
            self._collect_metric()
            return True
        if action_type == DiagnosisActionType.ENABLE_PROFILING:
            context = self._get_diagnosis_context() or {}
            return self._profiler_controller.enable(
                action,
                current_run_id=context.get("run_id", ""),
                current_rdzv_epoch=context.get("rdzv_epoch"),
            )

        logger.warning(f"Unsupported node action type: {action_type}")
        return False

    def _report_diagnosis_data(self, action):
        logger.info(
            "Report diagnosis data to master: %s (%s)",
            action.data_cls,
            action.reason,
        )
        try:
            response = self._master_client.report_diagnosis_data(
                action.data_cls,
                action.data_content,
                action.node_rank,
                report_id=action.report_id,
                data_type=action.data_type,
                schema_version=action.schema_version,
            )
        except Exception as error:
            logger.warning(f"Failed to report diagnosis data: {error}")
            if not action.is_expired():
                self._agent_context.enqueue_diagnosis_action(action)
            return False

        if response is None:
            logger.warning("Master did not acknowledge diagnosis data report.")
            if not action.is_expired():
                self._agent_context.enqueue_diagnosis_action(action)
            return False
        if not response.success:
            logger.warning(
                f"Master rejected diagnosis data report "
                f"{action.report_id or action.data_cls}."
            )
            return False
        return True

    def _report_event(self, action):
        labels = action.event_labels or {}
        self._master_client.report_event(
            event_type=action.event_type,
            instance=action.event_instance,
            action=action.event_action,
            msg=action.event_msg,
            labels=labels,
        )
        return True

    def _report_job_action(self, action):
        logger.info(
            f"Report job action to master for following process: "
            f"{action.action_type}."
        )
        self._master_client.report_action(action)
        return True

    def _reenqueue_event(self, action):
        time_diff = timestamp_diff_in_seconds(
            action.timestamp, datetime.now().timestamp()
        )
        expired_time_period = action.expired_time_period - time_diff
        if expired_time_period < 0:
            expired_time_period = 0
        action.update_timestamp(
            timestamp=datetime.now().timestamp(),
            expired_time_period=expired_time_period,
            executable_time_period=expired_time_period + 60,
        )
        self._agent_context.enqueue_diagnosis_action(action)
