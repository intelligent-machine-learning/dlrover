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
from datetime import datetime

from dlrover.python.common.constants import (
    Accelerators,
    GpuMetricEnum,
    NpuMetricEnum,
)
from dlrover.python.common.global_context import Context, DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
)
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisResult,
    JobHangWatermark,
)
from dlrover.python.diagnosis.common.diagnosis_action import NodeAction
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

_metric_context = JobMetricContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


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
        self._lock = threading.Lock()

    def collect_diagnosis_data(self, data: DiagnosisData):
        self._data_manager.store_data(data)

    def pre_check(self):
        logger.info("Start Diagnosis Manager to pre-check training...")
        pass

    def start_metric_collect(self):
        """
        create a XpuMetricMonitor instance based on worker XPU type
        start a thread to collect metrics data
        store the data into global JobMetricContext

        """
        logger.info(f"start {_dlrover_context.xpu_type} metric collector...")
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

        if _dlrover_context.xpu_type is Accelerators.ASCEND_NPU:
            self._metric_monitor = NpuMetricMonitor(
                job_name=self._job_name,
                metrics=[
                    NpuMetricEnum.NPU_UTIL,
                ],
            )
        else:
            self._metric_monitor = GpuMetricMonitor(
                job_name=self._job_name,
                metrics=[
                    GpuMetricEnum.GPU_UTIL,
                    GpuMetricEnum.GPU_TENSOR_UTIL,
                ],
            )

        if self._metric_monitor:
            self._metric_monitor.start()

    def stop_metric_collect(self):
        logger.info("stop metric collector...")
        if self._metric_monitor:
            self._metric_monitor.stop()

    def join_metric_collect(self):
        logger.info("join metric collector...")
        if self._metric_monitor:
            self._metric_monitor.join()

    def start_observing(self):
        logger.info("Start diagnosis manager training observation...")
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
            diag = threading.Thread(
                target=self._diagnose,
                name="diagnose_failures",
                daemon=True,
            )
            diag.start()
            if diag.is_alive():
                logger.info("_diagnose thread has started")

            diag_metric = threading.Thread(
                target=self._diagnose_metrics,
                name="diagnose_metrics",
                daemon=True,
            )
            diag_metric.start()
            if diag_metric.is_alive():
                logger.info("_diagnose_metrics thread has started")

        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis manager thread. Error: {e}"
            )

    def stop_observing(self):
        logger.info("Stop diagnosis manager training observation...")
        self._is_observing_started = False

    @staticmethod
    def check_tensor_drop_zero(duration):
        if duration > _metric_context.max_metric_records:
            duration = _metric_context.max_metric_records

        if _dlrover_context.xpu_type is Accelerators.ASCEND_NPU:
            metrics = _metric_context.backtrace_avg_metrics(
                NpuMetricEnum.NPU_UTIL, duration
            )
        else:
            metrics = _metric_context.backtrace_avg_metrics(
                GpuMetricEnum.GPU_TENSOR_UTIL, duration
            )

        if metrics is None:
            logger.warning(f"invalid metrics: {metrics}")
            return DiagnosisResult.DIAG_ERROR, 0, 0

        if len(metrics) < duration:
            logger.warning(
                f"waiting for tensor metrics: {len(metrics)}/{duration}"
            )
            return DiagnosisResult.DIAG_WAITING, 0, 0

        key_list = list(metrics.keys())
        key_list.sort(reverse=True)

        start_ts = key_list[0]
        end_ts = key_list[0]
        for key in key_list:
            end_ts = key
            if metrics[key] > JobHangWatermark.TENSOR_UTIL_LOW_WM:
                return DiagnosisResult.DIAG_HEALTHY, start_ts, end_ts

            duration = duration - 1
            if duration <= 0:
                break

        return DiagnosisResult.DIAG_HANG, start_ts, end_ts

    def _diagnose_metrics(self):
        logger.info("_diagnose_metrics thread is running...")
        while True:
            if not self._is_observing_started:
                logger.info(
                    f"stop _metric_diagnose thread due to "
                    f"{self._is_observing_started}"
                )
                break

            if (
                _dlrover_context.hang_downtime
                < DefaultValues.MIN_HANG_DOWNTIME
            ):
                hang_downtime = DefaultValues.MIN_HANG_DOWNTIME
            else:
                hang_downtime = _metric_context.hang_downtime
            result, start, end = self.check_tensor_drop_zero(hang_downtime)
            if result is DiagnosisResult.DIAG_HANG:
                start_dt = datetime.fromtimestamp(start).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                end_dt = datetime.fromtimestamp(end).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.warning(
                    f"detect job hang by tensor drop zero: "
                    f"{start_dt}-{end_dt}"
                )

                if _dlrover_context.hang_detection == 2:
                    self._job_context.enqueue_action(
                        NodeAction(
                            action_type=DiagnosisActionType.RESTART_WORKER,
                            instance=DiagnosisConstant.ANY_INSTANCE,
                        )
                    )

            time.sleep(DiagnosisConstant.METRIC_COLLECT_INTERVAL_SECS)

    def _diagnose(self):
        logger.info("_diagnose thread is running...")
        while True:
            if not self._is_observing_started:
                logger.info(
                    f"stop _diagnose thread due to "
                    f"{self._is_observing_started}"
                )
                break

            observed_problems = self._diagnostician.observe_training()
            action = self._diagnostician.resolve_problems(observed_problems)
            self._job_context.enqueue_action(action)

            time.sleep(
                DiagnosisConstant.MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS
            )
