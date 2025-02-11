from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    EventAction,
)
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    InferenceConfigKey,
    DiagnosisConstant,
)
from dlrover.python.common.constants import ErrorMonitorConstants
import json

class ResourceCollectObserver(Observer):
    """
    ResourceCollectObserver is to collect job resources
    """

    def __init__(self):
        super().__init__()
        self._monitor = ResourceMonitor()

    def observe(self) -> DiagnosisAction:
        error_logs = ""
        try:
            self._monitor.report_resource()
        except Exception as e:
            error_logs = f"{e}"

        if DiagnosisErrorConstant.GPU_LOST in error_logs:
            event_payload = {
                InferenceConfigKey.EVENT_TYPE: ErrorMonitorConstants.TYPE_WARN,  # noqa: E501
                InferenceConfigKey.EVENT_INSTANCE: f"{DiagnosisConstant.LOCAL_INSTANCE}",  # noqa: E501
                InferenceConfigKey.EVENT_ACTION: DiagnosisErrorConstant.GPU_LOST,
                InferenceConfigKey.EVENT_MSG: error_logs,
                InferenceConfigKey.EVENT_LABELS: json.dumps({}),
                InferenceConfigKey.EXPIRED_TIME_PERIOD: "120",
            },

            expired_time_period = (
                DiagnosisConstant.ACTION_EXPIRED_TIME_PERIOD_DEFAULT
            )
            if InferenceConfigKey.EXPIRED_TIME_PERIOD in event_payload:
                expired_time_period = int(
                    event_payload[InferenceConfigKey.EXPIRED_TIME_PERIOD]
                )
            executable_time_period = 0
            if InferenceConfigKey.EXECUTABLE_TIME_PERIOD in event_payload:
                executable_time_period = int(
                    event_payload[InferenceConfigKey.EXECUTABLE_TIME_PERIOD]
                )

            return EventAction(
                event_type=event_payload[InferenceConfigKey.EVENT_TYPE],
                event_instance=event_payload[
                    InferenceConfigKey.EVENT_INSTANCE
                ],
                event_action=event_payload[InferenceConfigKey.EVENT_ACTION],
                event_msg=event_payload[InferenceConfigKey.EVENT_MSG],
                event_labels=json.loads(
                    event_payload[InferenceConfigKey.EVENT_LABELS]
                ),
                expired_time_period=expired_time_period,
                executable_time_period=executable_time_period,
            )

        return NoAction()