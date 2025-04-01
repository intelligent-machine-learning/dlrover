from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)
from dlrover.python.common.constants import (
    EventReportConstants,
    DictKey,
)


class ResourceCollectDiagnostician(Diagnostician):
    """
    ResourceCollectDiagnostician is to collect and report resource
    to the master and handle the errors during collection
    """
    def __init__(self):
        super().__init__()
        self._monitor = ResourceMonitor().singleton_instance()

    def observe(self, **kwargs) -> DiagnosisObservation:
        error_logs = ""
        try:
            self._monitor.report_resource()
        except Exception as e:
            error_logs = f"{e}"

        if DiagnosisErrorConstant.GPU_LOST in error_logs:
            ob = DiagnosisObservation(DiagnosisErrorConstant.GPU_LOST)
            ob.extra_infos[DictKey.LOGS] = error_logs
            return ob
        else:
            return DiagnosisObservation()

    def resolve(
            self, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        if problem.observation == DiagnosisErrorConstant.GPU_LOST:
            return EventAction(
                event_type=EventReportConstants.TYPE_WARN,
                event_instance=f"{DiagnosisConstant.LOCAL_INSTANCE}",
                event_action=problem.observation,
                event_msg=problem.extra_infos.get(DictKey.LOGS, ""),
                event_labels={},
                expired_time_period=120,
            )
        else:
            return NoAction()
