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
from unittest import mock
from unittest.mock import patch

from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    EventReportConstants,
    NodeEnv,
    NodeType,
)
from dlrover.python.diagnosis.common.constants import (
    DiagnosisDataType,
    DiagnosisErrorConstant,
    EnvConfigKey,
    InferenceConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
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
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_failure_node_operator import (  # noqa: E501
    CheckFailureNodeOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.metrics_collection_operator import (  # noqa: E501
    MetricsCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.resource_collection_operator import (  # noqa: E501
    ResourceCollectionOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_gpu_errors_operator import (  # noqa: E501
    ResolveGPUErrorsOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_training_hang_operator import (  # noqa: E501
    ResolveTrainingHangOperator,
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
        self._master.stop()

    def test_check_training_hang_operator_find_intersection(self):
        test_metric: Dict[int, List[Tuple[int, bool]]] = {
            1: [(1, True), (2, False), (3, True), (4, True), (5, True)],
            2: [(1, True), (2, True), (3, True), (4, True), (5, False)],
            3: [(1, False), (2, True), (3, True), (4, True), (5, True)],
        }
        operator = CheckTrainingHangOperator(None)
        self.assertEqual(operator._get_hang_overlaps(test_metric), (-1, -1))

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
        self.assertEqual(operator._get_hang_overlaps(test_metric), (2, 1))

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
        self.assertEqual(operator._get_hang_overlaps(test_metric), (2, 2))

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
        self.assertEqual(operator._get_hang_overlaps(test_metric), (-1, -1))

    def test_check_training_hang_operator_is_hang(self):
        operator = CheckTrainingHangOperator(None)
        operator._get_hang_time_last_threshold = mock.MagicMock(return_value=0)

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

    def test_resolve_training_hang_operator(self):
        operator = ResolveTrainingHangOperator(None)
        input_infers = []
        result_infers = operator.infer(input_infers)
        self.assertEqual(
            result_infers,
            [
                Inference(
                    name=InferenceName.ACTION,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.NONE,
                )
            ],
        )

        input_infers.append(
            Inference(
                name=InferenceName.NODE,
                attribution=InferenceAttribute.ISORNOT,
                description=InferenceDescription.FAILURE,
                configs={
                    InferenceConfigKey.LOG_FILE: "test",
                    InferenceConfigKey.ERRORS: "error code is 123456",
                },
            )
        )
        result_infers = operator.infer(input_infers)
        self.assertEqual(
            result_infers,
            [
                Inference(
                    name=InferenceName.ACTION,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.NONE,
                )
            ],
        )

        input_infers.append(
            Inference(
                name=InferenceName.TRAINING,
                attribution=InferenceAttribute.IS,
                description=InferenceDescription.HANG,
            )
        )
        result_infers = operator.infer(input_infers)
        self.assertEqual(
            result_infers,
            [
                Inference(
                    name=InferenceName.ACTION,
                    attribution=InferenceAttribute.IS,
                    description=InferenceDescription.EVENT,
                    configs={
                        "event_type": EventReportConstants.TYPE_WARN,
                        "event_instance": EventReportConstants.JOB_INSTANCE,
                        "event_action": EventReportConstants.ACTION_HANG_WARN,
                        "event_msg": "",
                        "event_labels": "{}",
                    },
                )
            ],
        )

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

    def test_resource_collect_operator(self):
        error_log = "GPU is lost"

        res_collect_operator = ResourceCollectionOperator()
        res_collect_operator._monitor.report_resource = mock.MagicMock(
            side_effect=Exception(error_log)
        )

        res_collect_inf = Inference(
            name=InferenceName.WORKER,
            attribution=InferenceAttribute.COLLECT,
            description=InferenceDescription.RESOURCE,
        )
        self.assertTrue(res_collect_operator.is_compatible(res_collect_inf))

        gpu_error_inf = Inference(
            name=InferenceName.GPU,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.ERROR,
        )

        infs = res_collect_operator.infer([])
        self.assertEqual(len(infs), 1)

        self.assertTrue(is_same_inference(infs[0], gpu_error_inf))
        self.assertEqual(infs[0].configs[InferenceConfigKey.LOGS], error_log)

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

    def test_resolve_gpu_error_operator(self):
        error_log = DiagnosisErrorConstant.GPU_LOST

        operator = ResolveGPUErrorsOperator()
        gpu_error_inf = Inference(
            name=InferenceName.GPU,
            attribution=InferenceAttribute.IS,
            description=InferenceDescription.ERROR,
            configs={
                InferenceConfigKey.LOGS: error_log,
                InferenceConfigKey.ERRORS: DiagnosisErrorConstant.GPU_LOST,
            },
        )
        self.assertTrue(operator.is_compatible(gpu_error_inf))

        infs = operator.infer([gpu_error_inf])
        self.assertEqual(len(infs), 1)

        self.assertEqual(infs[0].name, InferenceName.ACTION)
        self.assertEqual(
            infs[0].configs[InferenceConfigKey.EVENT_TYPE],
            EventReportConstants.TYPE_WARN,
        )


if __name__ == "__main__":
    unittest.main()
