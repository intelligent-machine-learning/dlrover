# Copyright 2025 The DLRover Authors. All rights reserved.
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
import os
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import (
    NodeStatus,
    NodeType,
    DistributionStrategy,
    MthreadsGPUMetricEnum,
    Accelerators,
    GpuMetricEnum,
    PlatformType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.metric import GpuNodeMetric, GpuMetric
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    DiagnosisResult,
    DiagnosisActionType,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
    NodeAction,
)
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.diagnosis.diagnostician.node_failure import (
    FailureNodeDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.node_inconsistency import (
    NodeInconsistencyDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_failure import (  # noqa: E501
    ResourceCollectionFailureDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.training_hang import (
    TrainingHangDiagnostician,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.scheduler.kubernetes import k8sClient, K8sJobArgs
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.util.function_util import TimeoutException
from typing import Dict, List, Tuple, OrderedDict
from unittest import mock
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.common.constants import (
    DiagnosisDataType,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        self._master, self._addr = start_local_master()
        MasterClient._instance = build_master_client(self._addr, 1)

    def tearDown(self):
        os.environ.clear()
        self._master.stop()

    def test_diagnostician(self):
        diagnostician = Diagnostician(None)

        ob = diagnostician.observe()
        self.assertEqual(ob.observation, "unknown")

        actions = diagnostician.resolve(ob)
        self.assertTrue(isinstance(actions[0], NoAction))

        actions = diagnostician.diagnose()
        self.assertTrue(isinstance(actions[0], NoAction))

        diagnostician.resolve = MagicMock(side_effect=Exception())
        actions = diagnostician.diagnose()
        self.assertTrue(isinstance(actions[0], NoAction))

        with self.assertLogs(logger, level="ERROR") as log_capture:
            diagnostician.observe = MagicMock(side_effect=TimeoutException())
            diagnostician.diagnose()
            self.assertTrue(
                any("timeout" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

    def test_failure_node_diagnostician(self):
        diagnostician = FailureNodeDiagnostician()

        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 507035"

        ob = diagnostician.observe(log_file=file_path, errors=errors)
        self.assertEqual(ob.observation, DiagnosisErrorConstant.NODE_FAILED)

        ob = diagnostician.observe(log_file=file_path)
        self.assertFalse(ob)

        ob = diagnostician.observe(errors=errors)
        self.assertFalse(ob)

    def test_resource_collect_error_diagnostician(self):
        error_log = "GPU is lost"

        diagnostician = ResourceCollectionFailureDiagnostician()

        actions = diagnostician.diagnose(error_log=error_log)
        self.assertTrue(isinstance(actions[0], EventAction))

    def test_node_inconsistency_diagnostician(self):
        job_args = MagicMock()
        job_args.job_name = "test-job"
        job_args.namespace = "default"

        diagnostician = NodeInconsistencyDiagnostician(job_args)

        with patch.object(k8sClient, "singleton_instance") as mock_k8s_client:
            diagnostician._k8s_client = mock_k8s_client.return_value

            # Test basic functionality
            node1 = Node(
                node_type=NodeType.WORKER,
                node_id=0,
                rank_index=0,
                status=NodeStatus.RUNNING,
            )
            job_nodes = {NodeType.WORKER: {0: node1}}

            # Test case 1: No inconsistency
            mock_pod = MagicMock()
            mock_pod.status.phase = NodeStatus.RUNNING
            mock_pod.metadata.deletion_timestamp = None
            mock_pods = MagicMock()
            mock_pods.items = [mock_pod]
            mock_k8s_client.return_value.list_namespaced_pod.return_value = (
                mock_pods
            )

            observation = diagnostician.observe(job_nodes=job_nodes)
            self.assertIsNone(observation)

            # Test case 2: Multiple pods detected
            mock_pods.items = [mock_pod, mock_pod]  # Two pods
            mock_k8s_client.return_value.list_namespaced_pod.return_value = (
                mock_pods
            )

            observation = diagnostician.observe(job_nodes=job_nodes)
            self.assertIsNotNone(observation)
            self.assertEqual(observation.observation, "Repeated node")

            # Test resolve action
            actions = diagnostician.resolve(observation)
            self.assertTrue(isinstance(actions[0], EventAction))

            # Test resolve with correct observation format
            mock_problem = MagicMock()
            mock_problem.observation = DiagnosisErrorConstant.REPEATED_NODE
            actions = diagnostician.resolve(mock_problem)
            self.assertTrue(isinstance(actions[0], EventAction))

            # Test case 3: Empty nodes
            empty_nodes = {}
            observation = diagnostician.observe(job_nodes=empty_nodes)
            self.assertIsNone(observation)

    def test_training_hang_diagnostician_find_intersection(self):
        diagnostician = TrainingHangDiagnostician(None, None)

        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [(1, True), (2, False), (3, True), (4, True), (5, True)],
            2: [(1, True), (2, True), (3, True), (4, True), (5, False)],
            3: [(1, False), (2, True), (3, True), (4, True), (5, True)],
        }
        self.assertEqual(
            diagnostician._get_hang_overlaps(test_metric), (-1, -1)
        )

        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [
                (1, True),
                (2, False),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (7, True),
            ],
            2: [
                (1, True),
                (2, True),
                (3, True),
                (4, True),
                (5, False),
                (6, True),
                (7, True),
            ],
            3: [
                (1, False),
                (2, True),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (7, True),
            ],
        }
        self.assertEqual(diagnostician._get_hang_overlaps(test_metric), (2, 1))

        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [
                (1, True),
                (2, False),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (8, True),
            ],
            2: [
                (1, True),
                (2, True),
                (3, True),
                (4, True),
                (5, False),
                (6, True),
                (8, True),
            ],
            3: [
                (1, False),
                (2, True),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (8, True),
            ],
        }
        self.assertEqual(diagnostician._get_hang_overlaps(test_metric), (2, 2))

        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [
                (1, True),
                (2, False),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (8, False),
            ],
            2: [
                (1, True),
                (2, True),
                (3, True),
                (4, True),
                (5, False),
                (6, True),
                (8, True),
            ],
            3: [
                (1, False),
                (2, True),
                (3, True),
                (4, True),
                (5, True),
                (6, True),
                (8, True),
            ],
        }
        self.assertEqual(
            diagnostician._get_hang_overlaps(test_metric), (-1, -1)
        )

    def test_training_hang_diagnostician_is_hang_by_xpu_timer(self):
        job_args = JobArgs("local", "test", "test")
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        ob = diagnostician.observe()
        self.assertIsNone(ob)
        actions = diagnostician.resolve(ob)
        self.assertTrue(isinstance(actions[0], NoAction))

        diagnostician._get_hang_time_last_threshold = mock.MagicMock(
            return_value=0
        )

        # prepare test data
        # normal_metric, some_abnormal_metric, all_abnormal_metric = "", "", ""
        file_path = os.path.join(
            os.path.dirname(__file__),
            "data/xpu_timer/normal/xpu_timer_metric_0",
        )
        with open(file_path, "r", encoding="utf-8") as file:
            normal_metric = file.read()
        file_path = os.path.join(
            os.path.dirname(__file__),
            "data/xpu_timer/hang/xpu_timer_metric_some",
        )
        with open(file_path, "r", encoding="utf-8") as file:
            some_abnormal_metric = file.read()
        file_path = os.path.join(
            os.path.dirname(__file__),
            "data/xpu_timer/hang/xpu_timer_metric_all",
        )
        with open(file_path, "r", encoding="utf-8") as file:
            all_abnormal_metric = file.read()

        # test data: no worker hang
        w0_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=normal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w0_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=normal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w1_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=normal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        w1_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=normal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        test_data = [w0_t1, w1_t1, w0_t2, w1_t2]

        self.assertFalse(diagnostician.is_hang_by_xpu_timer_metric(test_data))
        test_data.clear()

        # test data0: 1 of 2 worker hang
        w0_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=some_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w0_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=some_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w1_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=some_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        w1_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=some_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        test_data = [w0_t1, w1_t1, w0_t2, w1_t2]

        self.assertFalse(diagnostician.is_hang_by_xpu_timer_metric(test_data))
        test_data.clear()

        # test data: 2 of 2 worker hang
        ts = int(time.time())
        w0_t1 = WorkerTrainingMetric(
            timestamp=ts,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w0_t2 = WorkerTrainingMetric(
            timestamp=ts + 1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w1_t1 = WorkerTrainingMetric(
            timestamp=ts,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        w1_t2 = WorkerTrainingMetric(
            timestamp=ts + 1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )

        data_mgr.store_data(w0_t1)
        data_mgr.store_data(w1_t1)
        data_mgr.store_data(w0_t2)
        data_mgr.store_data(w1_t2)
        ob = diagnostician.observe()
        self.assertEqual(
            ob.observation, DiagnosisErrorConstant.TRAINING_IS_HANG
        )

        actions = diagnostician.resolve(ob)
        self.assertTrue(isinstance(actions[-1], EventAction))

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._dlrover_context"
    )
    @patch("dlrover.python.diagnosis.diagnostician.training_hang._job_context")
    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._event_context"
    )
    def test_training_hang_diagnostician_is_hang_by_others(
        self, mock_event_context, mock_job_context, mock_dlrover_context
    ):
        job_args = JobArgs("local", "test", "test")
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # test observe
        # tensor_drop_zero: false + step_hang: false + ckpt_hang: false
        diagnostician._check_tensor_drop_zero = MagicMock(
            return_value=(DiagnosisResult.DIAG_HEALTHY, 0, 0)
        )
        mock_event_context.check_job_step_hang = MagicMock(return_value=False)
        mock_event_context.check_ckpt_hang = MagicMock(return_value=False)
        mock_event_context.check_event_block = MagicMock(return_value=False)
        self.assertIsNone(diagnostician.observe())

        # tensor_drop_zero: true + step_hang: false + ckpt_hang: false
        diagnostician._check_tensor_drop_zero = MagicMock(
            return_value=(DiagnosisResult.DIAG_HANG, 1, 2)
        )
        self.assertIsNone(diagnostician.observe())

        # tensor_drop_zero: true + step_hang: true + ckpt_hang: false
        mock_event_context.check_job_step_hang = MagicMock(return_value=True)
        ob = diagnostician.observe()
        self.assertIsNotNone(ob)
        self.assertEqual(
            ob.observation, DiagnosisErrorConstant.TRAINING_IS_HANG
        )

        # tensor_drop_zero: true + step_hang: false + ckpt_hang: true
        mock_event_context.check_job_step_hang = MagicMock(return_value=False)
        mock_event_context.check_ckpt_hang = MagicMock(return_value=True)
        ob = diagnostician.observe()
        self.assertIsNotNone(ob)
        self.assertEqual(
            ob.observation, DiagnosisErrorConstant.TRAINING_IS_HANG
        )

        mock_event_context.check_job_step_hang = MagicMock(return_value=False)
        mock_event_context.check_ckpt_hang = MagicMock(return_value=False)
        mock_event_context.check_event_block = MagicMock(return_value=True)
        ob = diagnostician.observe()
        self.assertIsNotNone(ob)
        self.assertEqual(
            ob.observation, DiagnosisErrorConstant.TRAINING_IS_HANG
        )

        # test resolve
        mock_nodes = {
            0: Node(node_type=NodeType.WORKER, node_id=0),
            1: Node(node_type=NodeType.WORKER, node_id=1),
            2: Node(node_type=NodeType.WORKER, node_id=2),
            3: Node(node_type=NodeType.WORKER, node_id=3),
        }
        mock_job_context.job_nodes_by_type = MagicMock(return_value=mock_nodes)
        total_nodes_num = len(mock_nodes)

        # hang detection level 0
        mock_dlrover_context.hang_detection = 0
        actions = diagnostician.resolve(ob)
        self.assertEqual(len(actions), total_nodes_num)
        for action in actions:
            self.assertTrue(isinstance(action, NodeAction))
            self.assertEqual(
                action.action_type, DiagnosisActionType.COLLECT_METRIC
            )
            self.assertTrue(action.instance in mock_nodes.keys())

        # hang detection level 1
        mock_dlrover_context.hang_detection = 1
        actions = diagnostician.resolve(ob)
        self.assertEqual(len(actions), total_nodes_num + 1)
        for i, action in enumerate(actions):
            if i < total_nodes_num:
                self.assertTrue(isinstance(action, NodeAction))
                self.assertEqual(
                    action.action_type, DiagnosisActionType.COLLECT_METRIC
                )
            else:
                self.assertTrue(isinstance(action, EventAction))

        # hang detection level 2
        mock_dlrover_context.hang_detection = 2
        actions = diagnostician.resolve(ob)
        self.assertEqual(len(actions), total_nodes_num + 1)
        for i, action in enumerate(actions):
            if i < total_nodes_num:
                self.assertTrue(isinstance(action, NodeAction))
                self.assertEqual(
                    action.action_type, DiagnosisActionType.COLLECT_METRIC
                )
            else:
                self.assertTrue(isinstance(action, NodeAction))
                self.assertEqual(
                    action.action_type, DiagnosisActionType.RESTART_WORKER
                )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_healthy(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for healthy mthreads GPU nodes."""
        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock healthy metrics - simulate tensor utilization above threshold
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())

        # Create metrics over time with high utilization
        for i in range(15):
            mock_metrics[current_time - i * 60] = (
                0.85  # High utilization (well above 0.001 threshold)
            )

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics
        mock_metric_context.max_metric_records = 100

        # Test with sufficient duration
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # Should return healthy result
        self.assertEqual(result, DiagnosisResult.DIAG_HEALTHY)
        self.assertIsNotNone(start_ts)
        self.assertIsNotNone(end_ts)

        # Verify correct metric enum was used
        mock_metric_context.backtrace_avg_metrics.assert_called_once_with(
            MthreadsGPUMetricEnum.GPU_TENSOR_UTIL, 10
        )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_low_utilization(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for mthreads GPU with low tensor utilization."""
        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock low utilization metrics
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())

        # Create metrics over time with low utilization (below threshold of 0.001)
        for i in range(15):
            mock_metrics[current_time - i * 60] = (
                0.0005  # Very low, below 0.001 threshold
            )

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics
        mock_metric_context.max_metric_records = 100

        # Test with sufficient duration
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # Should detect hanging due to low utilization
        self.assertEqual(result, DiagnosisResult.DIAG_HANG)
        self.assertIsNotNone(start_ts)
        self.assertIsNotNone(end_ts)

        # Verify correct metric enum was used
        mock_metric_context.backtrace_avg_metrics.assert_called_once_with(
            MthreadsGPUMetricEnum.GPU_TENSOR_UTIL, 10
        )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_no_metrics(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for mthreads GPU when no metrics available."""
        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock no metrics available
        mock_metric_context.backtrace_avg_metrics.return_value = None
        mock_metric_context.max_metric_records = 100

        # Test with no metrics
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # Should return error when no metrics
        self.assertEqual(result, DiagnosisResult.DIAG_ERROR)

        # Verify correct metric enum was used
        mock_metric_context.backtrace_avg_metrics.assert_called_once_with(
            MthreadsGPUMetricEnum.GPU_TENSOR_UTIL, 10
        )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_waiting_state(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for mthreads GPU in waiting state."""
        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock insufficient metrics (less than requested duration)
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())

        # Only 3 metric points (less than requested 10)
        for i in range(3):
            mock_metrics[current_time - i * 60] = (
                0.70  # Above threshold, but insufficient data
            )

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics
        mock_metric_context.max_metric_records = 100

        # Test with duration longer than available metrics
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # Should return waiting state
        self.assertEqual(result, DiagnosisResult.DIAG_WAITING)

        # Verify correct metric enum was used
        mock_metric_context.backtrace_avg_metrics.assert_called_once_with(
            MthreadsGPUMetricEnum.GPU_TENSOR_UTIL, 10
        )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_edge_threshold(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for mthreads GPU at edge of utilization threshold."""
        from dlrover.python.diagnosis.diagnostician.training_hang import (
            JobHangWatermark,
        )

        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock metrics at exactly the threshold
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())

        # Create metrics at threshold level
        for i in range(15):
            mock_metrics[current_time - i * 60] = (
                JobHangWatermark.TENSOR_UTIL_LOW_WM
            )

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics
        mock_metric_context.max_metric_records = 100

        # Test with sufficient duration
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # At threshold should still be considered hanging
        self.assertEqual(result, DiagnosisResult.DIAG_HANG)

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_mixed_utilization(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero for mthreads GPU with mixed utilization over time."""
        from dlrover.python.diagnosis.diagnostician.training_hang import (
            JobHangWatermark,
        )

        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Mock varying metrics over time
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())

        # Most recent metric is high (healthy), but older ones are low
        mock_metrics[current_time] = (
            JobHangWatermark.TENSOR_UTIL_LOW_WM + 0.01
        )  # High (recent)
        for i in range(1, 10):
            mock_metrics[current_time - i * 60] = (
                0.0005  # Low (older, below threshold)
            )

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics
        mock_metric_context.max_metric_records = 100

        # Test with sufficient duration
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(10)

        # Should return healthy because most recent metric is above threshold
        self.assertEqual(result, DiagnosisResult.DIAG_HEALTHY)

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._metric_context"
    )
    def test_check_tensor_drop_zero_mthreads_max_duration_limit(
        self, mock_metric_context
    ):
        """Test check_tensor_drop_zero respects max metric records limit."""
        job_args = JobArgs("local", "test", "test")
        job_args.xpu_type = Accelerators.MTHREADS_GPU
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(job_args, data_mgr)

        # Test with duration exceeding max_metric_records
        mock_metric_context.max_metric_records = 50
        excessive_duration = 150  # More than max_metric_records

        # Mock metrics
        mock_metrics = OrderedDict()
        current_time = int(datetime.now().timestamp())
        for i in range(60):  # More metrics than max_metric_records
            mock_metrics[current_time - i * 60] = 0.80  # High utilization

        mock_metric_context.backtrace_avg_metrics.return_value = mock_metrics

        # Test with excessive duration
        result, start_ts, end_ts = diagnostician._check_tensor_drop_zero(
            excessive_duration
        )

        # Should still work (duration gets clamped to max_metric_records)
        self.assertEqual(result, DiagnosisResult.DIAG_HEALTHY)

        # Verify duration was clamped to max_metric_records
        mock_metric_context.backtrace_avg_metrics.assert_called_once_with(
            MthreadsGPUMetricEnum.GPU_TENSOR_UTIL,
            50,  # Clamped to max_metric_records
        )

    @patch(
        "dlrover.python.diagnosis.diagnostician.training_hang._dlrover_context"
    )
    @patch(
        "dlrover.python.common.event.context.JobEventContext.check_job_step_hang",
    )
    def test_diagnose_metrics(
        self, mock_check_job_step_hang, mock_dlrover_context
    ):
        metric_context = JobMetricContext.singleton_instance()
        mock_check_job_step_hang.return_value = True

        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        mgr = DiagnosisMaster(job_args=args)
        diagnostician = TrainingHangDiagnostician(args, mgr)
        metric_context.clear_node_metrics()
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
            metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(10)[0],
            DiagnosisResult.DIAG_HANG,
        )
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(29)[0],
            DiagnosisResult.DIAG_HANG,
        )

        mock_dlrover_context.hang_downtime = 10
        mock_dlrover_context.hang_detection = 2

        diagnostician.diagnose()

        mgr._job_context.clear_job_nodes()
        mgr._job_context.clear_actions()
        metric_context.clear_node_metrics()

    def test_gpu_tensor_drop_zero(self):
        metric_context = JobMetricContext.singleton_instance()
        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        mgr = DiagnosisMaster(job_args=args)
        diagnostician = TrainingHangDiagnostician(args, mgr)
        metric_context.clear_node_metrics()

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
        metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(10)[0],
            DiagnosisResult.DIAG_WAITING,
        )

        for _ in range(10):
            ts = ts + 60
            metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(10)[0],
            DiagnosisResult.DIAG_HEALTHY,
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
            metric_context.add_node_metrics(ts, job_metrics)
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(10)[0],
            DiagnosisResult.DIAG_HANG,
        )
        self.assertEqual(
            diagnostician._check_tensor_drop_zero(29)[0],
            DiagnosisResult.DIAG_HANG,
        )
