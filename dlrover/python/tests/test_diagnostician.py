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

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import DiagnosisErrorConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
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
from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.util.function_util import TimeoutException
from typing import Dict, List, Tuple
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
        diagnostician = Diagnostician()

        ob = diagnostician.observe()
        self.assertEqual(ob.observation, "unknown")

        action = diagnostician.resolve(ob)
        self.assertTrue(isinstance(action, EventAction))

        action = diagnostician.diagnose()
        self.assertTrue(isinstance(action, EventAction))

        diagnostician.resolve = MagicMock(side_effect=Exception())
        action = diagnostician.diagnose()
        self.assertTrue(isinstance(action, NoAction))

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

        action = diagnostician.diagnose(error_log=error_log)
        self.assertTrue(isinstance(action, EventAction))

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
            action = diagnostician.resolve(observation)
            self.assertTrue(isinstance(action, EventAction))

            # Test resolve with correct observation format
            mock_problem = MagicMock()
            mock_problem.observation = DiagnosisErrorConstant.REPEATED_NODE
            action = diagnostician.resolve(mock_problem)
            self.assertTrue(isinstance(action, EventAction))

            # Test case 3: Empty nodes
            empty_nodes = {}
            observation = diagnostician.observe(job_nodes=empty_nodes)
            self.assertIsNone(observation)

    def test_training_hang_diagnostician_find_intersection(self):
        diagnostician = TrainingHangDiagnostician(None)

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

    def test_training_hang_diagnostician_is_hang(self):
        data_mgr = DiagnosisDataManager()

        diagnostician = TrainingHangDiagnostician(data_mgr)
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

        self.assertFalse(diagnostician.is_hang(test_data))
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

        self.assertFalse(diagnostician.is_hang(test_data))
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

        action = diagnostician.resolve(ob)
        self.assertTrue(isinstance(action, EventAction))
