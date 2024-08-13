from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.common.worker import WorkerContext
from typing import Dict
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from datetime import datetime
from dlrover.python.common.error import ProcessError
import json
from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.diagnose.inferencechain.inference_chain import InferenceChain
from dlrover.python.diagnose.common.inference_chain import (
    Inference,
    InferenceName,
    InferenceAttribute,
    InferenceDescription,
    include_inference,
)
from dlrover.python.diagnose.common.constants import (
    InferenceConfigKey,
    DiagnoseAction,
)
from dlrover.python.common.log import default_logger as logger


class DiagnoseAgent:
    def __init__(self, training_log_file: str, errors: str):
        self._client = MasterClient.singleton_instance()
        self._training_log_file = training_log_file
        self._errors = errors

    def diagnose_training(self, worker_context: WorkerContext) -> str:
        self._report_failure_to_master(worker_context.run_result.failures, worker_context.restart_count)
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
                f"[{worker_context.worker_spec.role}] Worker group {worker_context.run_result.state.name}. "
                f"{worker_context.remaining_failovers}/{worker_context.worker_spec.max_restarts}"
                f" attempts left; will restart worker group"
            )
            return DiagnoseAction.RESTART_WORKER
        else:
            return DiagnoseAction.RELAUNCH_WORKER

    def _report_failure_to_master(self, failures: Dict[int, ProcessFailure], restart_count: int):
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