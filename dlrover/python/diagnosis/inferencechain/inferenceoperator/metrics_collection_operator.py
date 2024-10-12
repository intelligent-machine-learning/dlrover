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
from dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector import (
    XpuTimerMetricsCollector,
)
from dlrover.python.elastic_agent.master_client import MasterClient


class MetricsCollectionOperator(InferenceOperator):
    """
    MetricsCollectionOperator is the operator to collect
    worker diagnosis metrics.
    """

    def __init__(self):
        super().__init__(None)
        self._xpu_timer_collector = XpuTimerMetricsCollector()
        self._client = MasterClient.singleton_instance()

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.WORKER
            and inference.attribution == InferenceAttribute.COLLECT
            and inference.description == InferenceDescription.METRICS
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
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

        return []
