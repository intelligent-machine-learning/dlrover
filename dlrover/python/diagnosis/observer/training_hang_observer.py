from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    EventAction,
)
from dlrover.python.diagnosis.common.observer import Observer
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceName,
    InferenceAttribute,
    InferenceDescription,
    is_inference_included,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import InferenceChain
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_training_hang_operator import CheckTrainingHangOperator
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.constants import ErrorMonitorConstants
import json
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    InferenceConfigKey,
)


class TrainingHangObserver(Observer):
    """
    TrainingHangObserver is to observe if a training job is hanged
    """

    def __init__(self, data_mgr):
        super().__init__()
        self.operators = [
            CheckTrainingHangOperator(data_mgr),
        ]

    def observe(self) -> DiagnosisAction:
        problem = Inference(
            InferenceName.TRAINING,
            InferenceAttribute.ISORNOT,
            InferenceDescription.HANG,
        )

        ic = InferenceChain([problem], self.operators)
        try:
            infs = ic.infer()
            hang_inf = Inference(
                InferenceName.TRAINING,
                InferenceAttribute.IS,
                InferenceDescription.HANG,
            )
            if is_inference_included(infs, hang_inf):
                event_payload = {
                    "event_type": ErrorMonitorConstants.TYPE_WARN,
                    "event_instance": ErrorMonitorConstants.JOB_INSTANCE,  # noqa: E501
                    "event_action": ErrorMonitorConstants.ACTION_HANG_WARN,  # noqa: E501
                    "event_msg": "",
                    "event_labels": json.dumps({}),
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
        except Exception as e:
            logger.error(f"Fail to observe training hang: {e}")
            return NoAction()