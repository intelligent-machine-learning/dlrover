from typing import List

from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import DiagnosisDataType
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)
from dlrover.python.diagnosis.common.constants import (
    InferenceConfigKey,
    DiagnosisErrorConstant,
)
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor


class ResourceCollectionOperator(InferenceOperator):
    """
    ResourceCollectionOperator is the operator to collect
    worker resources.
    """

    def __init__(self):
        super().__init__(None)
        self._monitor = ResourceMonitor()

    def is_compatible(self, inference: Inference) -> bool:
        if (
                inference.name == InferenceName.WORKER
                and inference.attribution == InferenceAttribute.COLLECT
                and inference.description == InferenceDescription.RESOURCE
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        error_logs = self._monitor.report_resource()
        if DiagnosisErrorConstant.GPU_LOST in error_logs:
            return [
                Inference(
                    name=InferenceName.GPU,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.ERROR,
                    configs={
                        InferenceConfigKey.LOGS: error_logs,
                        InferenceConfigKey.ERRORS: DiagnosisErrorConstant.GPU_LOST,
                    },
                ),
            ]

        return []
