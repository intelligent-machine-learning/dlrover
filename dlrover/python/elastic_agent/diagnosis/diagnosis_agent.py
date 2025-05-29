# Copyright 2024 The DLRover Authors. All rights reserved.
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
import threading
import time
from datetime import datetime

from dlrover.python.common import env_utils
from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisErrorConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NodeAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
)
from dlrover.python.diagnosis.datacollector.resource_collector import (
    ResourceCollector,
)
from dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector import (
    XpuTimerMetricsCollector,
)
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_error_diagnostician import (  # noqa: E501
    ResourceCollectErrorDiagnostician,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.training_event.config import is_dlrover_event_enabled


class DiagnosisAgent(Singleton, DiagnosisManager):
    def __init__(
        self,
        training_log_file="",
        errors="",
        node_rank=-1,
        local_world_size=0,
    ):
        self._client = MasterClient.singleton_instance()
        self._training_log_file = training_log_file
        self._errors = errors
        self._stopped = False
        self._agent_context = get_agent_context()

        DiagnosisManager.__init__(self, self._agent_context)
        # register diagnostician
        self.register_diagnostician(
            DiagnosisErrorConstant.NODE_FAILED, FailureNodeDiagnostician()
        )
        self.register_diagnostician(
            DiagnosisErrorConstant.RESOURCE_COLLECT_ERROR,
            ResourceCollectErrorDiagnostician(),
        )

        # register periodical diagnosis
        self.register_periodical_diagnosis(
            DiagnosisErrorConstant.RESOURCE_COLLECT_ERROR, 30
        )

        # register periodical data collector
        self.register_periodical_data_collector(XpuTimerMetricsCollector(), 60)
        self.register_periodical_data_collector(ResourceCollector(), 30)

        self._report_thread = None
        self._node_rank = node_rank
        self._local_world_size = local_world_size
        self._atorch_collector = AtorchEventCollector(
            local_world_size=local_world_size, retry_timeout=30
        )

        self.start()

        logger.info(
            "Initializing diagnosis agent with\n"
            f"training_log_file:    {self._training_log_file}\n"
            f"errors:               {self._errors}"
        )

    def update_config(
        self, training_log_file: str = "", errors: str = "", rank: int = -1
    ):
        if len(training_log_file) > 0:
            self._training_log_file = training_log_file
            logger.info(f"Update training_log_file: {training_log_file}")
        if len(errors) > 0:
            self._errors = errors
            logger.info(f"Update errors: {errors}")
        if rank >= 0:
            self._node_rank = rank
            logger.info(f"Update rank: {rank}")

    def start(self):
        self.start_diagnosis()
        self.start_data_collection()

        self._stopped = False

        self._report_thread = threading.Thread(
            target=self._periodically_report,
            name="periodically_reporter",
            daemon=True,
        )
        self._report_thread.start()

        if is_dlrover_event_enabled():
            self._atorch_collector.start_collectors()

    def stop(self):
        self._stopped = True
        if is_dlrover_event_enabled():
            self._atorch_collector.stop_collectors()

    def diagnose_training_failure(self) -> DiagnosisAction:
        self._report_failure_to_master(
            self._agent_context.run_result.failures,
            self._agent_context.restart_count,
        )
        ob = self.observe(
            DiagnosisErrorConstant.NODE_FAILED,
            log_file=self._training_log_file,
            errors=self._errors,
        )

        node_failed = ob.observation == DiagnosisErrorConstant.NODE_FAILED

        if self._agent_context.remaining_failovers > 0 and not node_failed:
            logger.info(
                f"[{self._agent_context.worker_spec.role}] Worker group "
                f"{self._agent_context.run_result.state.name}, "
                f"is failure node: {node_failed},"
                f"{self._agent_context.remaining_failovers}/"
                f"{self._agent_context.worker_spec.max_restarts} "
                f"attempts left; will restart worker group."
            )
            return NodeAction(
                node_id=env_utils.get_node_id(),
                node_type=env_utils.get_node_type(),
                instance=DiagnosisConstant.LOCAL_INSTANCE,
                action_type=DiagnosisActionType.RESTART_WORKER,
            )
        else:
            logger.info(
                f"[{self._agent_context.worker_spec.role}] Worker group "
                f"{self._agent_context.run_result.state.name}, "
                f"is failure node: {node_failed}, "
                f"no attempts("
                f"{self._agent_context.worker_spec.max_restarts}) "
                "left; will relaunch."
            )
            return NodeAction(
                node_id=env_utils.get_node_id(),
                node_type=env_utils.get_node_type(),
                instance=DiagnosisConstant.LOCAL_INSTANCE,
                action_type=DiagnosisActionType.RELAUNCH_WORKER,
            )

    def _report_failure_to_master(self, failures, restart_count):
        errors = {}
        if len(failures) == 0:
            logger.info("Skip failure report due to empty failures")
            return
        for rank, failure in failures.items():
            dt = str(datetime.fromtimestamp(int(failure.timestamp)))
            error = ProcessError(
                failure.local_rank, failure.exitcode, failure.message, dt
            )
            errors[rank] = error.__dict__
        error_data = json.dumps(errors)
        self._client.report_failures(
            error_data,
            restart_count,
            TrainingExceptionLevel.PROCESS_ERROR,
        )

    def _report_metric_to_master(self, agent_metric: WorkerTrainingMetric):
        self._client.report_diagnosis_agent_metrics(agent_metric)

    def send_heartbeat(self):
        try:
            ts = int(time.time())
            action = self._client.report_heart_beat(ts)
            self._agent_context.enqueue_diagnosis_action(action)
        except Exception as e:
            logger.warning(f"Fail to report a heartbeat: {e}")

    def _periodically_report(self):
        logger.info("Start diagnosis agent periodically reporter.")
        while True:
            if self._stopped:
                logger.info("Stop periodically reporter.")
                break
            self.send_heartbeat()
            time.sleep(
                DiagnosisConstant.AGENT_PERIODICALLY_REPORT_INTERVAL_SECS
            )
