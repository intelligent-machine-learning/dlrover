
from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    Observation,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import ObservationConstants

class FailureNodeObserver(Observer):
    """
    FailureNodeObserver is to observe if a node is failed
    """

    def __init__(self):
        super().__init__()

    def observe(self, log_file: str, errors: str) -> DiagnosisAction:
        # temp usage: express the env for specified error info
        # e.g.
        # export FAILURE_NODE_ERRORS="#error code is 12345# error code is
        # 23456# error code is 507035#"
        error_codes = errors.split("#")
        error_codes = [error_code.strip() for error_code in error_codes]

        collector = TrainingLogCollector(log_file, 5000)
        training_log = collector.collect_data()
        logs = training_log.logs
        if not logs or len(logs) == 0:
            logger.warning(f"fail to collect training logs from {log_file}")
            return NoAction()

        is_failure_node = False
        for log in logs:
            if is_failure_node:
                break
            for error in error_codes:
                if len(error) > 0 and "#" not in log and error in log:
                    logger.info(
                        f"Got #{error}# in {log}, set as failure node."
                    )
                    is_failure_node = True
                    break

        if is_failure_node:
            return Observation(
                observation=ObservationConstants.NODE_FAIL,
            )
        return NoAction()