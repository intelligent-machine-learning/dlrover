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
from typing import Dict, List

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    NoAction,
    NodeAction,
    Observation,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.diagnosis.diagnostician import Diagnostician
from dlrover.python.diagnosis.observer.failure_node_observer import FailureNodeObserver
from dlrover.python.diagnosis.observer.resource_collect_observer import ResourceCollectObserver
from dlrover.python.diagnosis.observer.metrics_collect_observer import MetricsCollectObserver


NodeFailed = "node_failed"
ResourceCollection = "resource_collection"
MetricsCollection = "metrics_collection"

class DiagnosisAgent(Singleton):
    def __init__(
        self,
        training_log_file="",
        errors="",
        rank=-1,
    ):
        self._client = MasterClient.singleton_instance()
        self._training_log_file = training_log_file
        self._errors = errors
        self._stopped = False

        self._diagnostician: Diagnostician = Diagnostician()
        self._diagnostician.register_observer(NodeFailed, FailureNodeObserver())
        self._diagnostician.register_observer(ResourceCollection, ResourceCollectObserver())
        self._diagnostician.register_observer(MetricsCollection, MetricsCollectObserver())


        # The key is the time interval in seconds
        self._observe_problems: Dict[int, List[str]] = {
            30: [
                ResourceCollection,
            ],
            60: [
                MetricsCollection,
            ],
        }
        self._accumulate_observe_time = 0

        self._agent_context = get_agent_context()
        self._diagnosis_thread = None
        self._report_thread = None
        self._rank = rank

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
        if len(errors) > 0:
            self._errors = errors
        if rank >= 0:
            self._rank = rank

    def start(self):
        self._stopped = False

        # start a async thread to diagnose periodically
        self._diagnosis_thread = threading.Thread(
            target=self._periodically_diagnosis,
            name="periodically_diagnostician",
            daemon=True,
        )
        self._diagnosis_thread.start()

        self._report_thread = threading.Thread(
            target=self._periodically_report,
            name="periodically_reporter",
            daemon=True,
        )
        self._report_thread.start()

    def stop(self):
        self._stopped = True

    def _get_observe_problems(self) -> List[str]:
        observe_problems: List[str] = []
        for time_period, problems in self._observe_problems.items():
            if (
                self._accumulate_observe_time > 0
                and self._accumulate_observe_time % time_period == 0
            ):
                observe_problems = observe_problems + problems
        return observe_problems

    def _periodically_diagnosis(self):
        logger.info("Start periodically diagnosis...")
        while True:
            if self._stopped:
                logger.info("Stop periodically diagnosis.")
                break
            observe_problems = self._get_observe_problems()
            for problem in observe_problems:
                action = self._diagnostician.observe(problem)
                if isinstance(action, Observation):
                    action = self._diagnostician.resolve(action)

                if not isinstance(action, NoAction):
                    self._agent_context.enqueue_diagnosis_action(action)

            if self._accumulate_observe_time > 600:
                self._accumulate_observe_time = 0

            time.sleep(
                DiagnosisConstant.AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS
            )
            self._accumulate_observe_time += (
                DiagnosisConstant.AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS
            )

    def diagnose_training_failure(self) -> NodeAction:
        self._report_failure_to_master(
            self._agent_context.run_result.failures,
            self._agent_context.restart_count,
        )
        ob = self._diagnostician.observe(NodeFailed, log_file=self._training_log_file, errors=self._errors)
        failure_node = False
        if isinstance(ob, Observation) and ob.node_failed():
            failure_node = True

        if self._agent_context.remaining_failovers > 0 and not failure_node:
            logger.info(
                f"[{self._agent_context.worker_spec.role}] Worker group "
                f"{self._agent_context.run_result.state.name}, "
                f"is failure node: {failure_node},"
                f"{self._agent_context.remaining_failovers}/"
                f"{self._agent_context.worker_spec.max_restarts} "
                f"attempts left; will restart worker group."
            )
            return NodeAction(
                action_type=DiagnosisActionType.RESTART_WORKER,
            )
        else:
            logger.info(
                f"[{self._agent_context.worker_spec.role}] Worker group "
                f"{self._agent_context.run_result.state.name}, "
                f"is failure node: {failure_node}, "
                f"no attempts("
                f"{self._agent_context.worker_spec.max_restarts}) "
                "left; will relaunch."
            )
            return NodeAction(
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
