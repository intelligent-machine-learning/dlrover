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
import threading
import time
import unittest
from unittest import mock
from unittest.mock import patch

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    EnvConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.common.diagnostician import Diagnostician
from dlrover.python.diagnosis.diagnostician.failure_node_diagnostician import (
    FailureNodeDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.metrics_collect_diagnostician import (  # noqa: E501
    MetricsCollectDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.resource_collect_diagnostician import (  # noqa: E501
    ResourceCollectDiagnostician,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.util.function_util import TimeoutException


class DiagnosticianTest(unittest.TestCase):
    def setUp(self):
        self._master, self._addr = start_local_master()
        MasterClient._instance = build_master_client(self._addr, 1)

    def tearDown(self):
        os.environ.clear()
        self._master.stop()

    @patch(
        "dlrover.python.diagnosis.common"
        ".diagnostician.Diagnostician.observe"
    )
    @patch(
        "dlrover.python.diagnosis.common"
        ".diagnostician.Diagnostician.resolve"
    )
    def test_diagnostician_exception(self, mock_resolve, mock_observe):
        mock_observe.side_effect = Exception("observe_error")
        mock_resolve.side_effect = Exception("resolve_error")

        context = get_agent_context()
        mgr = DiagnosisManager(context)

        name = "test"
        diagnostician = Diagnostician()
        mgr.register_diagnostician(name, diagnostician)

        ob = mgr.observe(name)
        self.assertTrue(len(ob.observation) == 0)

        action = mgr.resolve(name, ob)
        self.assertTrue(isinstance(action, NoAction))

        action = mgr.diagnose(name)
        self.assertTrue(isinstance(action, NoAction))

    @patch(
        "dlrover.python.diagnosis.common"
        ".diagnostician.Diagnostician.diagnose"
    )
    def test_diagnostician_start_periodical_diagnosis(self, mock_diagnose):
        context = get_agent_context()
        mgr = DiagnosisManager(context)
        diagnostician = Diagnostician()
        name = "test"
        mgr.register_diagnostician(name, diagnostician)
        mgr._periodical_diagnosis[name] = 0.1

        with self.assertLogs(logger, level="ERROR") as log_capture:
            thread = threading.Thread(
                target=mgr._start_periodical_diagnosis,
                name="test",
                args=(name,),
                daemon=True,
            )
            thread.start()
            time.sleep(0.2)
            self.assertTrue(context._diagnosis_action_queue.len() > 0)

            mock_diagnose.side_effect = TimeoutException()
            time.sleep(0.2)
            self.assertTrue(
                any("timeout" in msg for msg in log_capture.output),
                "Expected exception message not found in logs",
            )

            mock_diagnose.side_effect = Exception()
            time.sleep(0.2)
            self.assertTrue(
                any("Fail to diagnose" in msg for msg in log_capture.output),
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
        self.assertTrue(len(ob.observation) == 0)

        ob = diagnostician.observe(errors=errors)
        self.assertTrue(len(ob.observation) == 0)

    def test_resource_collect_diagnostician(self):
        error_log = "GPU is lost"

        diagnostician = ResourceCollectDiagnostician()
        diagnostician._monitor.report_resource = mock.MagicMock(
            side_effect=Exception(error_log)
        )

        action = diagnostician.diagnose()
        self.assertTrue(isinstance(action, EventAction))

    @patch(
        "dlrover.python.diagnosis.datacollector.xpu_timer_metric_collector"
        ".XpuTimerMetricsCollector.collect_data"
    )
    def test_collect_metrics_operator(self, mock_collector):
        mock_collector.return_value = "data"

        diagnostician = MetricsCollectDiagnostician()

        env_utils.set_env(EnvConfigKey.XPU_TIMER_PORT, 18889)
        env_utils.set_env(NodeEnv.NODE_ID, 1)
        env_utils.set_env(NodeEnv.NODE_TYPE, NodeType.WORKER)
        env_utils.set_env(NodeEnv.NODE_RANK, 1)
        ob = diagnostician.observe()
        self.assertEqual(len(ob.observation), 0)
