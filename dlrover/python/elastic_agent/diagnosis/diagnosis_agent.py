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

from dlrover.python.common import env_utils
from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    NodeAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    combine_inferences,
    is_inference_included,
)
from dlrover.python.diagnosis.inferencechain.coordinator import (
    coordinate_solutions,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    InferenceChain,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.operator import (  # noqa: E501
    get_training_failure_operators,
    get_worker_observe_operators,
    get_worker_resolve_operators,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import MasterClient


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
        # The key is the time interval in seconds
        self._observe_problems: Dict[int, List[Inference]] = {
            30: [
                Inference(
                    name=InferenceName.WORKER,
                    attribution=InferenceAttribute.COLLECT,
                    description=InferenceDescription.RESOURCE,
                ),
            ],
            60: [
                Inference(
                    name=InferenceName.WORKER,
                    attribution=InferenceAttribute.COLLECT,
                    description=InferenceDescription.METRICS,
                ),
            ],
        }
        self._accumulate_observe_time = 0

        self._observe_operators = get_worker_observe_operators()
        self._diagnosis_operators = get_worker_resolve_operators()
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

    def _get_observe_problems(self) -> List[Inference]:
        observe_problems: List[Inference] = []
        for time_period, infs in self._observe_problems.items():
            if (
                self._accumulate_observe_time > 0
                and self._accumulate_observe_time % time_period == 0
            ):
                observe_problems = observe_problems + infs
        return observe_problems

    def diagnose_problems(self, problems: List[Inference]) -> DiagnosisAction:
        conclusions: List[Inference] = []
        for problem in problems:
            if problem.configs is None:
                problem.configs = {}
            problem.configs[InferenceConfigKey.RANK] = str(self._rank)
            ic = InferenceChain([problem], self._diagnosis_operators)
            try:
                infs = ic.infer()
                if len(infs) > 0:
                    conclusions = combine_inferences(conclusions, infs)
            except Exception as e:
                logger.error(f"fail to diagnose observation {problem}: {e}")
        return coordinate_solutions(conclusions)

    def _observe(self, observe_problems: List[Inference]) -> List[Inference]:
        observations: List[Inference] = []
        for problem in observe_problems:
            ic = InferenceChain([problem], self._observe_operators)
            try:
                infs = ic.infer()
                if len(infs) > 0:
                    observations = combine_inferences(observations, infs)
            except Exception as e:
                logger.error(f"fail to observe problem {problem}: {e}")
        new_obs: List[Inference] = []
        for ob in observations:
            if not is_inference_included(observe_problems, ob):
                new_obs.append(ob)
        return new_obs

    def _diagnose_observations(
        self, observations: List[Inference]
    ) -> DiagnosisAction:
        if len(observations) == 0:
            return NoAction()
        conclusions: List[Inference] = []
        for ob in observations:
            ic = InferenceChain([ob], self._diagnosis_operators)
            try:
                infs = ic.infer()
                if len(infs) > 0:
                    conclusions = combine_inferences(conclusions, infs)
            except Exception as e:
                logger.error(f"fail to diagnose observation {ob}: {e}")
        return coordinate_solutions(conclusions)

    def _periodically_diagnosis(self):
        logger.info("Start periodically diagnosis...")
        while True:
            if self._stopped:
                logger.info("Stop periodically diagnosis.")
                break
            observe_problems = self._get_observe_problems()

            observations = self._observe(observe_problems)
            if len(observations) > 0:
                logger.debug(f"Observed problems: {observations}")
                action = self.diagnose_problems(observations)
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
        # check if the node is failed
        inference = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: self._training_log_file,
                InferenceConfigKey.ERRORS: self._errors,
            },
        )
        ic = InferenceChain([inference], get_training_failure_operators())
        infer_results = ic.infer()
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        failure_node = is_inference_included(infer_results, failure_inf)

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
                node_id=env_utils.get_node_id(),
                node_type=env_utils.get_node_type(),
                instance=DiagnosisConstant.LOCAL_INSTANCE,
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
