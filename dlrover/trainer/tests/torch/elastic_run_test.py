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
import unittest
from unittest.mock import patch

from dlrover.trainer.torch.elastic_run import (
    _check_dlrover_master_available,
    _check_to_use_dlrover_run,
    _elastic_config_from_args,
    _launch_dlrover_local_master,
    parse_args,
)

MC_PATH = "dlrover.python.elastic_agent.master_client.MasterClient"


class ElasticRunTest(unittest.TestCase):
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
            _check_to_use_dlrover_run("127.0.0.1:12345", 2, 3)

    def test_elastic_config_from_args(self):
        args = [
            "--network_check",
            "--comm_perf_test",
            "--auto_tunning",
            "--node_unit",
            "4",
            "--nnodes",
            "4",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        self.assertTrue(config.network_check)
        self.assertTrue(config.comm_perf_test)
        self.assertTrue(config.auto_tunning)
        self.assertEqual(config.node_unit, 4)
        self.assertEqual(config.rdzv_configs["node_unit"], 4)
        self.assertEqual(cmd, "/usr/local/bin/python")
        self.assertListEqual(cmd_args, ["-u", "test.py", "--batch_size", "16"])
