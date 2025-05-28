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

import copy
import threading
import time
import unittest
from datetime import datetime
from typing import List
from unittest import mock
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import (
    Accelerators,
    GpuMetricEnum,
    NodeStatus,
    NodeType,
    NpuMetricEnum,
    PlatformType,
    PreCheckStatus,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.metric import GpuMetric, GpuNodeMetric
from dlrover.python.common.metric.monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
)
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisDataType,
    DiagnosisResult,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    DiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    is_training_hanged,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.diagnosis.precheck_operator import (
    PreCheckOperator,
    PreCheckResult,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.scheduler.kubernetes import K8sJobArgs
from dlrover.python.util.function_util import TimeoutException

_metric_context = JobMetricContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


class DiagnosisMasterTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DiagnosisDataManager(1)
        log1 = TrainingLog(0)
        mgr.store_data(log1)
        time.sleep(0.01)
        log2 = TrainingLog(0)
        mgr.store_data(log2)

        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 2)

        time.sleep(1.5)
        log3 = TrainingLog(0)
        mgr.store_data(log3)
        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 1)

    def test_diagnosis_master_api(self):
        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        monitor = GpuMetricMonitor(
            job_name=args.job_name,
            metrics=[
                GpuMetricEnum.GPU_UTIL,
                GpuMetricEnum.GPU_TENSOR_UTIL,
            ],
        )
        mgr = DiagnosisMaster(job_args=args)
        mgr.new_metric_monitor(monitor)
        mgr.start_metric_collect()
        mgr.stop_metric_collect()

        args.xpu_type = Accelerators.ASCEND_NPU
        monitor = NpuMetricMonitor(
            job_name=args.job_name,
            metrics=[
                NpuMetricEnum.NPU_UTIL,
            ],
        )
        mgr = DiagnosisMaster(job_args=args)
        mgr.new_metric_monitor(monitor)
        mgr.start_metric_collect()
        mgr.stop_metric_collect()

        mgr = DiagnosisMaster(job_args=args)
        mgr.pre_check()
        mgr.start_observing()
        mgr.stop_observing()

    def test_diagnosis_master(self):
        mgr = DiagnosisMaster()
        problems: List[Inference] = [
            Inference(
                InferenceName.TRAINING,
                InferenceAttribute.ISORNOT,
                InferenceDescription.HANG,
            )
        ]
        mgr._diagnostician.register_training_problems(problems)
        self.assertEqual(len(mgr._diagnostician._training_problems), 1)

        data_mgr = DiagnosisDataManager(10000)
        operator = CheckTrainingHangOperator(data_mgr)
        mgr._diagnostician.register_observers([operator])
        self.assertEqual(len(mgr._diagnostician._observers), 1)

        data = DiagnosisData(
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content="XPU_TIMER_COMMON_HANG",
        )
        data_mgr.store_data(data)

        # mock training hang
        mgr._diagnostician._observers[0].is_hang = mock.MagicMock(
            return_value=True
        )

        # observe training problems
        observed_problems = mgr._diagnostician.observe_training()
        self.assertTrue(is_training_hanged(observed_problems[0]))

        # explore solutions to observed problems
        action = mgr._diagnostician.resolve_problems(observed_problems)
        self.assertEqual(action.action_type, DiagnosisActionType.NONE)

    def test_gpu_tensor_drop_zero(self):
        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        mgr = DiagnosisMaster(job_args=args)
        _metric_context.clear_node_metrics()

        job_metrics = {}
        metric = GpuNodeMetric()
        for i in range(8):
            metric.node_metrics[i] = GpuMetric()
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_FREE_MEM, 0)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_USED_MEM, 80)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_UTIL, 99.5)
            metric.node_metrics[i].set_metric(
                GpuMetricEnum.GPU_TENSOR_UTIL, 0.307
            )
        metric.update_avg_metrics()
        job_metrics["worker-1"] = copy.deepcopy(metric)
        job_metrics["worker-2"] = copy.deepcopy(metric)
        job_metrics["worker-3"] = copy.deepcopy(metric)
        job_metrics["worker-4"] = copy.deepcopy(metric)

        ts = int(datetime.now().timestamp())
        _metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            mgr.check_tensor_drop_zero(10)[0], DiagnosisResult.DIAG_WAITING
        )

        for _ in range(10):
            ts = ts + 60
            _metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            mgr.check_tensor_drop_zero(10)[0], DiagnosisResult.DIAG_HEALTHY
        )

        job_metrics = {}
        metric = GpuNodeMetric()
        for i in range(8):
            metric.node_metrics[i] = GpuMetric()
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_FREE_MEM, 0)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_USED_MEM, 80)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_UTIL, 99.5)
            metric.node_metrics[i].set_metric(
                GpuMetricEnum.GPU_TENSOR_UTIL, 0.0002
            )
        metric.update_avg_metrics()
        job_metrics["worker-1"] = copy.deepcopy(metric)
        job_metrics["worker-2"] = copy.deepcopy(metric)
        job_metrics["worker-3"] = copy.deepcopy(metric)
        job_metrics["worker-4"] = copy.deepcopy(metric)

        for _ in range(30):
            ts = ts + 60
            _metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            mgr.check_tensor_drop_zero(10)[0], DiagnosisResult.DIAG_HANG
        )
        self.assertEqual(
            mgr.check_tensor_drop_zero(29)[0], DiagnosisResult.DIAG_HANG
        )

    @patch(
        "dlrover.python.common.event.context.JobEventContext"
        ".check_job_step_hang",
    )
    def test_diagnose_metrics(self, mock_check_job_step_hang):
        mock_check_job_step_hang.return_value = True

        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        mgr = DiagnosisMaster(job_args=args)
        _metric_context.clear_node_metrics()
        mgr._job_context.clear_job_nodes()
        mgr._job_context.clear_actions()

        job_metrics = {}
        metric = GpuNodeMetric()
        for i in range(8):
            metric.node_metrics[i] = GpuMetric()
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_FREE_MEM, 0)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_USED_MEM, 80)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_UTIL, 99.5)
            metric.node_metrics[i].set_metric(
                GpuMetricEnum.GPU_TENSOR_UTIL, 0.0002
            )
        metric.update_avg_metrics()
        job_metrics["worker-1"] = copy.deepcopy(metric)
        job_metrics["worker-2"] = copy.deepcopy(metric)
        job_metrics["worker-3"] = copy.deepcopy(metric)
        job_metrics["worker-4"] = copy.deepcopy(metric)

        ts = int(datetime.now().timestamp())
        for _ in range(30):
            ts = ts + 60
            _metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            mgr.check_tensor_drop_zero(10)[0], DiagnosisResult.DIAG_HANG
        )
        self.assertEqual(
            mgr.check_tensor_drop_zero(29)[0], DiagnosisResult.DIAG_HANG
        )

        _dlrover_context.hang_downtime = 10
        _dlrover_context.hang_detection = 2
        mgr._is_observing_started = True
        mgr._is_observing_paused = False

        def stop_diagnose_metric():
            time.sleep(1)
            mgr._is_observing_started = False

        thread = threading.Thread(
            target=stop_diagnose_metric,
            daemon=True,
        )
        thread.start()
        mgr._diagnose_metrics(interval=1)

        mgr._is_observing_started = True
        mgr._is_observing_paused = False
        node = Node(
            node_type=NodeType.WORKER,
            node_id=3,
            rank_index=0,
            name="worker-0",
            status=NodeStatus.RUNNING,
        )
        mgr._job_context.update_job_node(node)

        thread = threading.Thread(
            target=stop_diagnose_metric,
            daemon=True,
        )
        thread.start()
        mgr._diagnose_metrics(interval=1)

        action = mgr._job_context.next_action(DiagnosisConstant.ANY_INSTANCE)
        self.assertEqual(
            action.action_type, DiagnosisActionType.RESTART_WORKER
        )
        self.assertEqual(action.instance, DiagnosisConstant.ANY_INSTANCE)
        self.assertEqual(action.node_id, 3)

        mgr._is_observing_started = False
        mgr._is_observing_paused = False
        mgr._job_context.clear_job_nodes()
        mgr._job_context.clear_actions()
        _metric_context.clear_node_metrics()

    @patch(
        "dlrover.python.master.diagnosis.diagnosis_master"
        ".get_pre_check_timeout"
    )
    def test_pre_check(self, mock_get_pre_check_timeout):
        mock_get_pre_check_timeout.return_value = 2
        job_context = get_job_context()
        mgr = DiagnosisMaster()
        original_get_pre_check_status = mgr._job_context.get_pre_check_status

        mgr.pre_check()
        self.assertEqual(job_context._action_queue.len(), 0)

        _dlrover_context = Context.singleton_instance()
        _dlrover_context.pre_check_operators = [
            ("dlrover.python.tests.test_diagnosis_master", "TestOperator")
        ]
        mgr._job_context.get_pre_check_status = MagicMock(
            return_value=PreCheckStatus.CHECKING
        )
        mgr.get_pre_check_operators = MagicMock(return_value=[TestOperator()])
        try:
            mgr.pre_check()
            self.fail()
        except TimeoutException:
            pass
        mgr._job_context.get_pre_check_status = original_get_pre_check_status

        _dlrover_context.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                True,
            )
        ]
        mgr.pre_check()
        self.assertEqual(
            job_context.get_pre_check_status(), PreCheckStatus.PASS
        )

        _dlrover_context.pre_check_operators = []
        mgr.pre_check()
        self.assertEqual(
            job_context.get_pre_check_status(), PreCheckStatus.DISABLED
        )

        mgr._job_context.get_pre_check_status = MagicMock(
            return_value=PreCheckStatus.PASS
        )
        mgr.pre_check()
        mgr._job_context.get_pre_check_status = original_get_pre_check_status


class TestOperator(PreCheckOperator):
    @classmethod
    def get_retry_interval_secs(cls) -> int:
        raise TimeoutException()

    def check(self, *args, **kwargs) -> PreCheckResult:
        return PreCheckResult(1, "test", [Node("worker", 0)])

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]
