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
import unittest
from unittest import mock

from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common import env_utils
from dlrover.python.common.constants import RendezvousName
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    NoAction,
    NodeAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.elastic_agent.context import get_agent_context
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
        self._master, self.addr = start_local_master()
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
        os.environ.clear()
        self._master.stop()

    def test_diagnose_agent(self):
        agent = DiagnosisAgent.singleton_instance()
        agent.stop()

        agent._accumulate_observe_time = 0
        problems = agent._get_observe_problems()
        self.assertEqual(len(problems), 0)

        agent._accumulate_observe_time = 30
        problems = agent._get_observe_problems()
        self.assertEqual(len(problems), 1)

        agent._accumulate_observe_time = 60
        problems = agent._get_observe_problems()
        self.assertEqual(len(problems), 2)

    def test_diagnose_training(self):
        file = "data/training.log"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)

        errors = "error code is 11111"

        agent = DiagnosisAgent.singleton_instance()
        agent.update_config(file_path, errors)

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

        context = get_agent_context()

        self.assertTrue("worker_spec", context.to_string())

        context.update_context(
            worker_spec=spec,
            remaining_failovers=2,
            restart_count=3,
            run_result=run_result,
        )

        action = agent.diagnose_training_failure()
        self.assertEqual(
            action.action_type, DiagnosisActionType.RESTART_WORKER
        )

        agent._errors = "error code is 507035"
        action = agent.diagnose_training_failure()
        self.assertEqual(
            action.action_type, DiagnosisActionType.RELAUNCH_WORKER
        )

        agent._errors = "error code is 11111"
        context.remaining_failovers = 0
        action = agent.diagnose_training_failure()
        self.assertEqual(
            action.action_type, DiagnosisActionType.RELAUNCH_WORKER
        )

        agent._errors = " #"
        context.remaining_failovers = 2
        action = agent.diagnose_training_failure()
        self.assertEqual(
            action.action_type, DiagnosisActionType.RESTART_WORKER
        )

        agent.stop()

    def test_worker_training_metric(self):
        test = WorkerTrainingMetric(
            data_content="test123",
            node_id=env_utils.get_node_id(),
            node_type=env_utils.get_node_type(),
            node_rank=env_utils.get_node_rank(),
            is_final_result=True,
        )

        test_str = test.to_json()
        self.assertTrue('"data_content": "test123"' in test_str)

        test_new = WorkerTrainingMetric.from_json(test_str)
        self.assertEqual(test_new.timestamp, test.timestamp)
        self.assertEqual(test_new.data_content, test.data_content)
        self.assertEqual(test_new.data_type, test.data_type)
        self.assertEqual(test_new.is_final_result, test.is_final_result)

        test_new = globals().get("WorkerTrainingMetric").from_json(test_str)
        self.assertEqual(test_new.timestamp, test.timestamp)
        self.assertEqual(test_new.data_content, test.data_content)
        self.assertEqual(test_new.data_type, test.data_type)
        self.assertEqual(test_new.is_final_result, test.is_final_result)

        test_new = globals().get(test.__class__.__name__).from_json(test_str)
        self.assertEqual(test_new.timestamp, test.timestamp)
        self.assertEqual(test_new.data_content, test.data_content)
        self.assertEqual(test_new.data_type, test.data_type)
        self.assertEqual(test_new.is_final_result, test.is_final_result)

    def test_send_heartbeat(self):
        agent = DiagnosisAgent.singleton_instance("", "")
        context = agent._agent_context
        agent._client.report_heart_beat = mock.MagicMock(
            returnValue=NoAction()
        )

        agent.send_heartbeat()
        self.assertTrue(
            context._diagnosis_action_queue.next_action().action_type,
            DiagnosisActionType.NONE,
        )

        agent._client.report_heart_beat = mock.MagicMock(
            returnValue=NodeAction(
                node_id=0,
                node_type="worker",
                action_type=DiagnosisActionType.RESTART_WORKER,
            )
        )
        agent.send_heartbeat()
        self.assertTrue(
            context._diagnosis_action_queue.next_action().action_type,
            DiagnosisActionType.RESTART_WORKER,
        )

        agent._client.report_heart_beat = mock.MagicMock(
            side_effect=[Exception]
        )
        agent.send_heartbeat()
        self.assertTrue(
            context._diagnosis_action_queue.next_action().action_type,
            DiagnosisActionType.NONE,
        )

    def test_async_thread(self):
        DiagnosisConstant.AGENT_PERIODICALLY_DIAGNOSIS_INTERVAL_SECS = 1
        DiagnosisConstant.AGENT_PERIODICALLY_REPORT_INTERVAL_SECS = 1
        agent = DiagnosisAgent("", "")
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("periodically_diagnostician", active_threads_name)
        self.assertIn("periodically_reporter", active_threads_name)

        agent.stop()
        time.sleep(2)
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertNotIn("periodically_diagnostician", active_threads_name)
        self.assertNotIn("periodically_reporter", active_threads_name)

        agent.start()
        time.sleep(2)
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("periodically_diagnostician", active_threads_name)
        self.assertIn("periodically_reporter", active_threads_name)

        agent.stop()


if __name__ == "__main__":
    unittest.main()
