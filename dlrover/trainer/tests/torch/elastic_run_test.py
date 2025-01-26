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

import socket
import telnetlib
import threading
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
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

    def tearDown(self):
        self._master.stop

    @patch(f"{MC_PATH}.report_failures")
    def test_launch_local_master(self, some_func):
        available = _check_dlrover_master_available("", timeout=3)
        self.assertFalse(available)
        handler, addr = _launch_dlrover_local_master("", "test", 1)
        available = _check_dlrover_master_available(addr, timeout=15)
        self.assertTrue(available)

        def mock_telent(*args, **kwargs):
            raise socket.gaierror("Mock gaierror")

        old_telnet = telnetlib.Telnet
        telnetlib.Telnet = mock_telent
        some_func.return_value = True
        with self.assertRaises(socket.gaierror):
            _check_dlrover_master_available(addr)
        telnetlib.Telnet = old_telnet
        handler.close()

    def test_check_to_use_dlrover_run(self):
        use_dlrover_run = _check_to_use_dlrover_run("", 1, 3)
        self.assertFalse(use_dlrover_run)
        with self.assertRaises(ValueError):
            _check_to_use_dlrover_run("", 2, 3)
        with self.assertRaises(ValueError):
            _check_to_use_dlrover_run("127.0.0.1:23456", 2, 3)

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
        client = MasterClient.singleton_instance()

        # pre-check success
        client.get_pre_check_result = MagicMock(return_value=True)
        wait_pre_check()

        # pre-check fail
        client.get_pre_check_result = MagicMock(return_value=False)

        def set_pre_check_success():
            time_to_set_success = time.time()
            while True:
                if time.time() - time_to_set_success > 1:
                    client.get_pre_check_result = MagicMock(return_value=True)
                    break
                time.sleep(0.1)

        start = time.time()
        threading.Thread(target=set_pre_check_success).start()
        wait_pre_check()
        self.assertTrue(time.time() - start > 0.5)
