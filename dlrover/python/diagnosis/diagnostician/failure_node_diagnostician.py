from dlrover.python.diagnosis.diagnostician.diagnostician import Diagnostician
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    ObservationAction,
    NodeAction,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    Observation,
    DiagnosisConstant,
    DiagnosisActionType,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.common import env_utils


class FailureNodeDiagnostician(Diagnostician):
    """
    FailureNodeDiagnostician is to observe and resolve the failure node problem
    """
    def __init__(self):
        super().__init__()
        self._agent_context = get_agent_context()

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
            return ObservationAction(
                observation=Observation.NODE_FAILED,
            )
        return NoAction()

    def resolve(self, problem: DiagnosisAction, **kwargs) -> DiagnosisAction:
        if not isinstance(problem, ObservationAction):
            return NoAction()

        problem.__class__ = ObservationAction
        if self._agent_context.remaining_failovers > 0 and not problem.node_failed():
            logger.info(
                f"[{self._agent_context.worker_spec.role}] Worker group "
                f"{self._agent_context.run_result.state.name}, "
                f"is failure node: {problem.node_failed()},"
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
                f"is failure node: {problem.node_failed()}, "
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
