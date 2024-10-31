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

from torch.distributed.elastic.multiprocessing.errors import ProcessFailure

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionConstants,
    DiagnosisConstant,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction, DiagnosisNodeAction
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    combine_inferences,
    is_inference_included,
)
from dlrover.python.diagnosis.inferencechain.coordinate_inferences import (
    coordinate_inferences,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    InferenceChain,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.operator import (  # noqa: E501
    get_training_failure_operators,
    get_worker_diagnosis_operators,
    get_worker_observe_operators,
)
from dlrover.python.elastic_agent.common.worker_context import WorkerContext
from dlrover.python.elastic_agent.master_client import MasterClient


class DiagnosisAgent(Singleton):
    def __init__(
        self,
        training_log_file: str,
        errors: str,
        worker_context: WorkerContext,
    ):
        self._client = MasterClient.singleton_instance()
        self._training_log_file = training_log_file
        self._errors = errors
        self._stopped = False
        self._observe_problems: List[Inference] = [
            Inference(
                name=InferenceName.WORKER,
                attribution=InferenceAttribute.COLLECT,
                description=InferenceDescription.METRICS,
            ),
        ]
        self._observe_operators = get_worker_observe_operators()
        self._diagnosis_operators = get_worker_diagnosis_operators()
        self._worker_context = worker_context
        self._diagnosis_thread = None
        self._report_thread = None

        self.start()

        logger.info(
            "Initializing diagnosis agent with\n"
            f"training_log_file:    {self._training_log_file}\n"
            f"errors:               {self._errors}"
        )

    def start(self):
        self._stopped = False

        # start a async thread to diagnose periodically
        self._diagnosis_thread = threading.Thread(
            target=self._periodically_diagnosis,
            name="periodically_diagnosis",
            daemon=True,
        )
        self._diagnosis_thread.start()

        self._report_thread = threading.Thread(
            target=self._periodically_report,
            name="diagnosis_reporter",
            daemon=True,
        )
        self._report_thread.start()

    def stop(self):
        self._stopped = True

    def diagnose_problems(
        self, problems: List[Inference]
    ) -> List[DiagnosisAction]:
        conclusions: List[Inference] = []
        for problem in problems:
            ic = InferenceChain([problem], self._diagnosis_operators)
            try:
                infs = ic.infer()
                if len(infs) > 0:
                    conclusions = combine_inferences(conclusions, infs)
            except Exception as e:
                logger.error(f"fail to diagnose observation {problem}: {e}")
        return coordinate_inferences(conclusions)

    def _observe(self) -> List[Inference]:
        observations: List[Inference] = []
        for problem in self._observe_problems:
            ic = InferenceChain([problem], self._observe_operators)
            try:
                infs = ic.infer()
                if len(infs) > 0:
                    observations = combine_inferences(observations, infs)
            except Exception as e:
                logger.error(f"fail to observe problem {problem}: {e}")
        return observations

    def _periodically_diagnosis(self):
        logger.info("Start periodically diagnosis...")
        while True:
            if self._stopped:
                logger.info("Stop periodically diagnosis.")
                break

            observations = self._observe()
            if len(observations) > 0:
                logger.info(f"Observed problems: {observations}")
                self.diagnose_problems(observations)

            time.sleep(
                DiagnosisConstant.AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS
            )

    def diagnose_training_failure(self) -> DiagnosisAction:
        self._report_failure_to_master(
            self._worker_context.run_result.failures,
            self._worker_context.restart_count,
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

        if self._worker_context.remaining_failovers > 0 and not failure_node:
            logger.info(
                f"[{self._worker_context.worker_spec.role}] Worker group "
                f"{self._worker_context.run_result.state.name}, "
                f"is failure node: {failure_node},"
                f"{self._worker_context.remaining_failovers}/"
                f"{self._worker_context.worker_spec.max_restarts} "
                f"attempts left; will restart worker group."
            )
            return DiagnosisNodeAction(
                action=DiagnosisActionConstants.RESTART_WORKER,
                instance=DiagnosisConstant.LOCAL_INSTANCE,
            )
        else:
            logger.info(
                f"[{self._worker_context.worker_spec.role}] Worker group "
                f"{self._worker_context.run_result.state.name}, "
                f"is failure node: {failure_node}, "
                f"no attempts({self._worker_context.worker_spec.max_restarts}) "
                "left; will relaunch."
            )
            return DiagnosisNodeAction(
                action=DiagnosisActionConstants.RELAUNCH_WORKER,
                instance=DiagnosisConstant.LOCAL_INSTANCE,
            )

    def _report_failure_to_master(
        self, failures: Dict[int, ProcessFailure], restart_count: int
    ):
        errors = {}
        if len(failures) == 0:
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
            actions = self._client.report_heart_beat(ts)
            for action in actions:
                self._worker_context.enqueue_diagnose_action(action)
        except Exception as e:
            logger.warning(f"fail to report a heartbeat: {e}")

    def _periodically_report(self):
        logger.info("Start diagnosis agent reporter.")
        while True:
            self.send_heartbeat()
            time.sleep(15)
