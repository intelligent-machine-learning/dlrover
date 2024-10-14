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
import unittest
from typing import Dict, List, Tuple
from unittest.mock import patch

from diagnosis.datacollector.training_log_collector import TrainingLogCollector
from diagnosis.datacollector.xpu_timer_metric_collector import \
    XpuTimerMetricsCollector
from dlrover.python.diagnosis.common.constants import (
    DiagnosisDataType,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv, NodeType
from dlrover.python.diagnosis.common.constants import (
    EnvConfigKey,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
    is_same_inference,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    InferenceChain,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_failure_node_operator import (  # noqa: E501
    CheckFailureNodeOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.metrics_collection_operator import (  # noqa: E501
    MetricsCollectionOperator,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master


class InferenceChainTest(unittest.TestCase):
    def setUp(self):
        self._master, self._addr = start_local_master()
        MasterClient._instance = build_master_client(self._addr, 1)

    def tearDown(self):
        os.environ.clear()

    def test_check_training_hang_operator_find_intersection(self):
        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [(1, True), (2, False), (3, True), (4, True), (5, True)],
            2: [(1, True), (2, True), (3, True), (4, True), (5, False)],
            3: [(1, False), (2, True), (3, True), (4, True), (5, True)],
        }
        operator = CheckTrainingHangOperator(None)
        self.assertEqual(
            operator._find_hang_intersection(test_metric), (-1, -1)
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
        operator = CheckTrainingHangOperator(None)
        self.assertEqual(operator._find_hang_intersection(test_metric), (2, 1))

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
        operator = CheckTrainingHangOperator(None)
        self.assertEqual(operator._find_hang_intersection(test_metric), (2, 2))

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
        operator = CheckTrainingHangOperator(None)
        self.assertEqual(
            operator._find_hang_intersection(test_metric), (-1, -1)
        )

    def test_check_training_hang_operator_is_hang(self):
        operator = CheckTrainingHangOperator(None)
        test_data = []

        # prepare test data
        normal_metric, some_abnormal_metric, all_abnormal_metric = "", "", ""
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

        self.assertFalse(operator.is_hang(test_data))
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

        self.assertFalse(operator.is_hang(test_data))
        test_data.clear()

        # test data: 2 of 2 worker hang
        w0_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w0_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=0,
            node_type="worker",
            node_rank=0,
        )
        w1_t1 = WorkerTrainingMetric(
            timestamp=1,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        w1_t2 = WorkerTrainingMetric(
            timestamp=2,
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=all_abnormal_metric,
            node_id=1,
            node_type="worker",
            node_rank=1,
        )
        test_data = [w0_t1, w1_t1, w0_t2, w1_t2]

        self.assertTrue(operator.is_hang(test_data))
        test_data.clear()

    def test_check_training_hang_operator(self):
        # no data
        operator = CheckTrainingHangOperator(None)
        inf = Inference(
            name=InferenceName.TRAINING,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.HANG,
        )
        self.assertTrue(operator.is_compatible(inf))

        results = operator.infer([inf])
        self.assertEqual(
            results[0],
            Inference(
                name=InferenceName.TRAINING,
                attribution=InferenceAttribute.NOT,
                description=InferenceDescription.HANG,
            ),
        )

    def test_check_failure_node_operator(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        operator = CheckFailureNodeOperator()
        inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: file_path,
                InferenceConfigKey.ERRORS: "error code is 507035",
            },
        )
        self.assertTrue(operator.is_compatible(inf))

        results = operator.infer([inf])
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(is_same_inference(results[0], failure_inf))

        #########################################################
        inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: file_path,
                InferenceConfigKey.ERRORS: "error code is 123456",
            },
        )

        results = operator.infer([inf])
        not_failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.NOT,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(is_same_inference(results[0], not_failure_inf))

    def test_inference_chain(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)
        inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.ISORNOT,
            description=InferenceDescription.FAILURE,
            configs={
                InferenceConfigKey.LOG_FILE: file_path,
                InferenceConfigKey.ERRORS: "error code is 507035",
            },
        )

        operators = [CheckFailureNodeOperator()]
        ic = InferenceChain([inf], operators)
        results = ic.infer()
        failure_inf = Inference(
            name=InferenceName.NODE,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.FAILURE,
        )
        self.assertTrue(is_same_inference(results[0], failure_inf))

    @patch(
        "dlrover.python.diagnosis.datacollector.training_log_collector"
        ".read_last_n_lines"
    )
    def test_log_collect(self, mock_file_util):
        mock_file_util.return_value = [
            "test0",
            "DLRover agent started with:",
            "test1",
        ]
        training_log_collector = TrainingLogCollector(
            log_file="test", n_line=3
        )
        self.assertTrue(training_log_collector.is_enabled())
        result = training_log_collector.collect_data()
        self.assertTrue("test0" not in result.logs)
        self.assertTrue("test1" in result.logs)

    def test_xpu_timer_metric_collect(self):
        collector = XpuTimerMetricsCollector()
        self.assertFalse(collector.is_enabled())

        env_utils.set_env(EnvConfigKey.XPU_TIMER_PORT, 18889)
        collector = XpuTimerMetricsCollector()
        self.assertTrue(collector.is_enabled())

        self.assertEqual(collector.collect_data(), "")

        file = "data/xpu_timer/xpu_timer_metric_single"
        file_path = os.path.join(os.path.dirname(__file__), file)
        with open(file_path, "r", encoding="utf-8") as file:
            test_metrics = file.read()
        result = collector._preprocess_metrics(test_metrics)
        self.assertTrue(result)
        if "#" in result or "exposer" in result:
            self.fail()

        env_utils.set_env(NodeEnv.NODE_ID, 1)
        env_utils.set_env(NodeEnv.NODE_TYPE, NodeType.WORKER)
        env_utils.set_env(NodeEnv.NODE_RANK, 1)
        agent_xpu_metric = WorkerTrainingMetric(
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=result,
            node_id=env_utils.get_node_id(),
            node_type=env_utils.get_node_type(),
            node_rank=env_utils.get_node_rank(),
        )
        self.assertEqual(
            agent_xpu_metric.data_type,
            DiagnosisDataType.XPU_TIMER_METRIC,
        )
        self.assertEqual(agent_xpu_metric.data_content, result)
        self.assertEqual(agent_xpu_metric.node_id, 1)
        self.assertEqual(agent_xpu_metric.node_type, NodeType.WORKER)
        self.assertEqual(agent_xpu_metric.node_rank, 1)
        self.assertTrue(agent_xpu_metric.timestamp > 0)

    @patch(
        "dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector"
        ".XpuTimerMetricsCollector.collect_data"
    )
    def test_collect_metrics_operator(self, mock_collector):
        mock_collector.return_value = "data"
        operator = MetricsCollectionOperator()
        inf = Inference(
            name=InferenceName.WORKER,
            attribution=InferenceAttribute.COLLECT,
            description=InferenceDescription.METRICS,
        )
        self.assertTrue(operator.is_compatible(inf))

        env_utils.set_env(EnvConfigKey.XPU_TIMER_PORT, 18889)
        env_utils.set_env(NodeEnv.NODE_ID, 1)
        env_utils.set_env(NodeEnv.NODE_TYPE, NodeType.WORKER)
        env_utils.set_env(NodeEnv.NODE_RANK, 1)
        infs = operator.infer([])
        self.assertEqual(len(infs), 0)


if __name__ == "__main__":
    unittest.main()
