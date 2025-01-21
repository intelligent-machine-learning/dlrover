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

import os
import threading
import time

from dlrover.python.common.constants import (
    Accelerators,
    GpuMetricEnum,
    NpuMetricEnum,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
)
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.inference_chain import (
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import Inference
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_training_hang_operator import (  # noqa: E501
    ResolveTrainingHangOperator,
)
from dlrover.python.master.diagnosis.diagnosis import Diagnostician
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.node.job_context import get_job_context


class DiagnosisManager:
    """
    DiagnosisManager is to manage all diagnosis issues in a training job

    """

    def __init__(self, job_name=None):
        self._is_observing_started = False
        self._data_manager: DiagnosisDataManager = DiagnosisDataManager(600)
        self._diagnostician: Diagnostician = Diagnostician(self._data_manager)
        self._job_context = get_job_context()
        self._job_name = job_name
        self._metric_monitor = None
        self._xpu_type = ""
        self._lock = threading.Lock()

    def collect_diagnosis_data(self, data: DiagnosisData):
        self._data_manager.store_data(data)

    def collect_xpu_info(self, xpu_type):
        with self._lock:
            if not xpu_type:
                return
            if not self._xpu_type:
                logger.info(f"the XPU accelerator is {xpu_type}")
                self._xpu_type = xpu_type
                self.start_metric_collect()

    def get_xpu_info(self):
        with self._lock:
            return self._xpu_type

    def pre_check(self):
        logger.info("Start Diagnosis Manager to pre-check training...")
        pass

    def start_metric_collect(self):
        """
        create a XpuMetricMonitor instance based on worker XPU type
        start a thread to collect metrics data
        store the data into global JobMetricContext

        """
        logger.info(f"start {self._xpu_type} metric collector...")
        url = os.getenv("DLROVER_METRIC_URL", "")
        if url == "":
            logger.warning("no GPU metrics url defined, stop metric collector")
            return
        token = os.getenv("DLROVER_METRIC_TOKEN", "")
        if token == "":
            logger.warning(
                "no GPU metrics token defined, stop metric collector"
            )
            return

        if self._xpu_type is Accelerators.ASCEND_NPU:
            self._metric_monitor = NpuMetricMonitor(
                job_name=self._job_name,
                metrics=[
                    NpuMetricEnum.NPU_UTIL,
                ],
            )
        elif self._xpu_type is Accelerators.NVIDIA_GPU:
            self._metric_monitor = GpuMetricMonitor(
                job_name=self._job_name,
                metrics=[
                    GpuMetricEnum.GPU_UTIL,
                    GpuMetricEnum.GPU_TENSOR_UTIL,
                ],
            )
        else:
            logger.warning(
                f"invalid XPU {self._xpu_type}, stop metric collector"
            )
            return

        if self._metric_monitor:
            self._metric_monitor.start()

    def stop_metric_collect(self):
        logger.info(f"stop {self._xpu_type} metric collector...")
        if self._metric_monitor:
            self._metric_monitor.stop()

    def join_metric_collect(self):
        logger.info(f"join {self._xpu_type} metric collector...")
        if self._metric_monitor:
            self._metric_monitor.join()

    def start_observing(self):
        logger.info("Start Diagnosis Manager to observing training...")
        self._is_observing_started = True

        self._diagnostician.register_training_problems(
            [
                Inference(
                    InferenceName.TRAINING,
                    InferenceAttribute.ISORNOT,
                    InferenceDescription.HANG,
                )
            ]
        )
        self._diagnostician.register_observers(
            [CheckTrainingHangOperator(self._data_manager)]
        )
        self._diagnostician.register_resolvers(
            [ResolveTrainingHangOperator(self._data_manager)]
        )

        try:
            thread = threading.Thread(
                target=self._diagnose,
                name="diagnose_failures",
                daemon=True,
            )
            thread.start()
            if thread.is_alive():
                logger.info("Diagnosis Manager is started")
        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis manager thread. Error: {e}"
            )

    def stop_observing(self):
        logger.info("Stop Diagnosis Manager to observing training.")
        self._is_observing_started = False

    def _diagnose(self):
        logger.info("Start to diagnose failures for observing.")
        while True:
            if not self._is_observing_started:
                logger.info("Stop to diagnose failures for observing.")
                break

            observed_problems = self._diagnostician.observe_training()
            action = self._diagnostician.resolve_problems(observed_problems)
            self._job_context.enqueue_action(action)

            time.sleep(
                DiagnosisConstant.MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS
            )
