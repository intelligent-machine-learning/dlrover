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
from unittest.mock import patch

from diagnosis.common.diagnosis_data import WorkerTrainingMetric
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv, NodeType, RendezvousName
from dlrover.python.common.worker import WorkerContext
from dlrover.python.diagnosis.common.constants import (
    DiagnosisAction,
    DiagnosisDataType,
    EnvConfigKey,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)
from dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector import (
    XpuTimerMetricsCollector,
)
from dlrover.python.elastic_agent.diagnosis.diagnosis_agent import (
    DiagnosisAgent,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.training import (
    ElasticLaunchConfig,
    _create_worker_spec,
)
from dlrover.python.tests.test_utils import start_local_master


class TestDiagnosisAgent(unittest.TestCase):
    def setUp(self):
        self.master_proc, self.addr = start_local_master()
        MasterClient._instance = build_master_client(self.addr, 1)
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="test",
            monitor_interval=0.1,
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)

    def tearDown(self):
        pass

    def test_diagnose_training(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 11111"
        agent = DiagnosisAgent.singleton_instance(file_path, errors)

        spec = _create_worker_spec(
            node_rank=0,
            rdzv_name=RendezvousName.ELASTIC_TRAINING,
            config=self.config,
            entrypoint="echo",
            args=[],
        )

        run_result = RunResult(
            state=WorkerState(
                WorkerState.UNHEALTHY,
            ),
            failures={},
        )
        wc = WorkerContext(
            worker_spec=spec,
            remaining_failovers=2,
            restart_count=3,
            run_result=run_result,
        )

        action = agent.diagnose_training_failure(wc)
        self.assertEqual(action, DiagnosisAction.RESTART_WORKER)

        agent._errors = "error code is 507035"
        action = agent.diagnose_training_failure(wc)
        self.assertEqual(action, DiagnosisAction.RELAUNCH_WORKER)

        agent._errors = "error code is 11111"
        wc.remaining_failovers = 0
        action = agent.diagnose_training_failure(wc)
        self.assertEqual(action, DiagnosisAction.RELAUNCH_WORKER)

        agent._errors = " #"
        wc.remaining_failovers = 2
        action = agent.diagnose_training_failure(wc)
        self.assertEqual(action, DiagnosisAction.RESTART_WORKER)

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

        file = "data/xpu_timer_metrics"
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


if __name__ == "__main__":
    unittest.main()
