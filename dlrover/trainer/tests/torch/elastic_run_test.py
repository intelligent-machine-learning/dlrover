# Copyright 2023 The DLRover Authors. All rights reserved.
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
import socket
import threading
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    JobConstant,
    NodeEnv,
    PreCheckStatus,
)
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.training import ElasticLaunchConfig
from dlrover.python.tests.test_utils import start_local_master
from dlrover.trainer.torch.elastic_run import (
    _check_dlrover_master_available,
    _check_to_use_dlrover_run,
    _elastic_config_from_args,
    _launch_dlrover_local_master,
    parse_args,
    wait_pre_check,
)

MC_PATH = "dlrover.python.elastic_agent.master_client.MasterClient"


class ElasticRunTest(unittest.TestCase):
    def setUp(self):
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 1)
        self._pre_check_interval_ori = JobConstant.PRE_CHECK_WAIT_SECS
        JobConstant.PRE_CHECK_WAIT_SECS = 1

    def tearDown(self):
        self._master.stop()
        os.environ.clear()
        JobConstant.PRE_CHECK_WAIT_SECS = self._pre_check_interval_ori

    def test_launch_local_master(self):
        handler, addr = _launch_dlrover_local_master("test:1234", "test")
        self.assertEqual(addr, "test:1234")
        self.assertIsNotNone(handler)
        handler.close()

        handler, addr = _launch_dlrover_local_master("", "test")
        self.assertTrue("127.0.0.1" in addr)
        self.assertIsNotNone(handler)
        handler.close()

    @patch("dlrover.trainer.torch.elastic_run.telnetlib.Telnet")
    def test_check_dlrover_master_available(self, mock_telnet):
        self.assertFalse(_check_dlrover_master_available("", timeout=0))
        self.assertFalse(_check_dlrover_master_available("test123", timeout=0))

        mock_telnet.return_value = MagicMock()
        self.assertTrue(
            _check_dlrover_master_available("test123:1234", timeout=0)
        )

        mock_telnet.side_effect = ConnectionRefusedError()
        self.assertFalse(
            _check_dlrover_master_available("test123:1234", timeout=0)
        )

        mock_telnet.side_effect = socket.gaierror
        self.assertFalse(
            _check_dlrover_master_available("test123:1234", timeout=0)
        )

    @patch("dlrover.trainer.torch.elastic_run._check_dlrover_master_available")
    @patch("dlrover.trainer.torch.elastic_run._launch_dlrover_local_master")
    def test_check_to_use_dlrover_run(
        self,
        mock_launch_local_master,
        mock_check_dlrover_master,
    ):
        job_name = "test"

        # 1) no dist master 2) dist mode
        mock_check_dlrover_master.return_value = False
        try:
            self.assertFalse(_check_to_use_dlrover_run(job_name))
            self.fail()
        except Exception:
            pass

        # 1) no dist master 2) standalone mode + no local master 3) node 0
        mock_check_dlrover_master.return_value = False
        mock_launch_local_master.return_value = None, "127.0.0.1:8000"
        self.assertFalse(_check_to_use_dlrover_run(job_name, True))

        # 1) with master address 2) node 0
        mock_check_dlrover_master.return_value = True
        self.assertTrue(_check_to_use_dlrover_run(job_name, True))

        # 1) with master address 2) node 1
        env_utils.set_env(NodeEnv.WORKER_RANK, "1")
        self.assertTrue(_check_to_use_dlrover_run(job_name, True))

        # 1) no master address 2) node 1
        mock_check_dlrover_master.return_value = False
        try:
            self.assertFalse(_check_to_use_dlrover_run(job_name, True))
            self.fail()
        except Exception:
            pass

        # 1) no dist master 2) standalone mode 3) node 1
        try:
            self.assertFalse(_check_to_use_dlrover_run(job_name))
            self.fail()
        except Exception:
            pass

    def test_elastic_config_from_args(self):
        args = [
            "--precheck",
            "1",
            "--auto_tunning",
            "--node_unit",
            "4",
            "--nnodes",
            "4",
            "--training_port",
            "1000",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        self.assertEqual(config.precheck, 1)
        self.assertTrue(config.network_check)
        self.assertFalse(config.comm_perf_test)
        self.assertTrue(config.auto_tunning)
        self.assertEqual(config.node_unit, 4)
        self.assertEqual(config.rdzv_configs["node_unit"], 4)
        self.assertEqual(config.training_port, 1000)
        self.assertTrue("bin/python" in cmd)
        self.assertListEqual(cmd_args, ["-u", "test.py", "--batch_size", "16"])

    @patch(f"{MC_PATH}.get_elastic_run_config")
    def test_elastic_config_from_master_1(self, mock_func):
        mock_func.return_value = {
            "network_check": "True",
            "comm_perf_test": "True",
            "auto_tunning": "True",
            "auto_config": "True",
            "exclude_straggler": "True",
            "save_at_breakpoint": "True",
        }
        args = [
            "--training_port",
            "1000",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        self.assertTrue(config.network_check)
        self.assertTrue(config.comm_perf_test)
        self.assertTrue(config.auto_tunning)
        self.assertTrue(config.auto_config)
        self.assertTrue(config.exclude_straggler)
        self.assertTrue(config.save_at_breakpoint)

    def test_elastic_config_from_master_2(self):
        MasterClient._instance.get_elastic_run_config = mock.MagicMock(
            side_effect=Exception()
        )

        args = [
            "--training_port",
            "1000",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        self.assertFalse(config.network_check)

    def test_wait_pre_check(self):
        test_config = ElasticLaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1
        )
        client = MasterClient.singleton_instance()

        # pre-check success
        client.get_pre_check_result = MagicMock(
            return_value=PreCheckStatus.PASS
        )
        client.report_pre_check_status = MagicMock(return_value=True)
        wait_pre_check(test_config)

        # pre-check fail
        client.get_pre_check_result = MagicMock(
            return_value=PreCheckStatus.FAIL
        )

        def set_pre_check_success():
            time_to_set_success = time.time()
            while True:
                if time.time() - time_to_set_success > 1:
                    client.get_pre_check_result = MagicMock(
                        return_value=PreCheckStatus.PASS
                    )
                    break
                time.sleep(0.1)

        start = time.time()
        threading.Thread(target=set_pre_check_success).start()
        wait_pre_check(test_config)
        self.assertTrue(time.time() - start > 0.5)

    @patch(
        "dlrover.trainer.torch.elastic_run.MasterClient.singleton_instance",
        return_value=None,
    )
    def test_wait_pre_check_with_none_client(self, mock_client):
        with self.assertRaises(RuntimeError):
            wait_pre_check(
                ElasticLaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
            )
            mock_client.assert_called_once()
