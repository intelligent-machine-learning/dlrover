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
from datetime import datetime
from typing import Dict

from torch.distributed.elastic.multiprocessing.errors import ProcessFailure

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.worker import WorkerContext
from dlrover.python.diagnose.common.constants import (
    DiagnoseAction,
    InferenceConfigKey,
)
from dlrover.python.diagnose.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    include_inference,
)
from dlrover.python.diagnose.inferencechain.inference_chain import (
    InferenceChain,
)
from dlrover.python.elastic_agent.master_client import MasterClient


class DiagnoseAgent:
    def __init__(self, training_log_file: str, errors: str):
        self._client = MasterClient.singleton_instance()
        self._training_log_file = training_log_file
        self._errors = errors

    def diagnose_training(self, worker_context: WorkerContext) -> str:
        self._report_failure_to_master(
            worker_context.run_result.failures, worker_context.restart_count
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
        ic = InferenceChain([inference])
        infer_results = ic.infer()
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        failure_node = include_inference(infer_results, failure_inf)

        if worker_context.remaining_failovers > 0 and not failure_node:
            logger.info(
                f"[{worker_context.worker_spec.role}] Worker group "
                f"{worker_context.run_result.state.name}. "
                f"{worker_context.remaining_failovers}/"
                f"{worker_context.worker_spec.max_restarts}"
                f" attempts left; will restart worker group"
            )
            return DiagnoseAction.RESTART_WORKER
        else:
            return DiagnoseAction.RELAUNCH_WORKER

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
