from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector import (
    XpuTimerMetricsCollector,
)
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import DiagnosisDataType

class MetricsCollectObserver(Observer):
    """
    MetricsCollectObserver is to collect job runtime metrics
    """

    def __init__(self):
        super().__init__()
        self._xpu_timer_collector = XpuTimerMetricsCollector()
        self._client = MasterClient.singleton_instance()

    def observe(self) -> DiagnosisAction:
        xpu_timer_metric = self._xpu_timer_collector.collect_data()
        if xpu_timer_metric:
            agent_xpu_metric = WorkerTrainingMetric(
                data_type=DiagnosisDataType.XPU_TIMER_METRIC,
                data_content=xpu_timer_metric,
                node_id=env_utils.get_node_id(),
                node_type=env_utils.get_node_type(),
                node_rank=env_utils.get_node_rank(),
            )
            self._client.report_diagnosis_agent_metrics(agent_xpu_metric)

        return NoAction()