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

import datetime

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.diagnosis import K8sPodData, DiagnosisDataType
from dlrover.python.master.diagnosis.diagnosis_data import DataManager
from dlrover.python.master.diagnosis.inferencechain.common import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    InferenceOperator,
)


class CheckPodPendingOperator(InferenceOperator):
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def is_compatible(self, inference: Inference) -> bool:
        if (
            inference.name == InferenceName.POD
            and inference.attribution == InferenceAttribute.ISORNOT
            and inference.description == InferenceDescription.PENDING
        ):
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        # data = K8sPodData(0, pods)
        # DiagnosisManager.singleton_instance().collect_diagnosis_data(DiagnosisDataType.K8SPODDATA, data)
        pod_data = self.data_manager.get_data(DiagnosisDataType.K8SPODDATA)
        if pod_data is None or len(pod_data) == 0:
            logger.info('[PodPendingChecker] No pod data collected yet.')
            return []
        k8s_pod_data = pod_data[-1]
        if not isinstance(k8s_pod_data, K8sPodData):
            logger.info('[PodPendingChecker] data is not instance of K8sPodData.')
            return []

        pods = k8s_pod_data.pods
        logger.info(f'[PodPendingChecker] {len(pods)} pods collected at {pod_data[len(pod_data) - 1].timestamp}')
        # check pod pending time
        for pod in pods:
            if pod.status.phase == 'Pending':
                if pod.status.conditions is None or len(pod.status.conditions) == 0:
                    logger.info(f'[PodPendingChecker] Pod {pod.metadata.name} has no conditions.')
                    continue
                start_time = pod.status.conditions[-1].last_transition_time
                time_difference = (datetime.now() - start_time).total_seconds() / 60
                logger.info(f'[PodPendingChecker] Pod {pod.metadata.name} is pending for {time_difference} minutes')
                if time_difference > 15:
                    # TODO: add inference and do restart for pod
                    logger.info(f'[PodPendingChecker] TODO: restart Pod {pod.metadata.name}')
        return []
