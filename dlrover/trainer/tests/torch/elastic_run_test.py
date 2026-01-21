# Copyright 2026 The DLRover Authors. All rights reserved.
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
from argparse import Namespace
from unittest import mock
from unittest.mock import ANY, MagicMock, patch

import dlrover
from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    JobConstant,
    NodeEnv,
    PreCheckStatus,
)
from dlrover.python.common.enums import FailoverStrategy
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.dynamic_failover import (
    DynamicAgentFailoverExtension,
    AgentFailureInfo,
)
from dlrover.python.elastic_agent.torch.training import ElasticLaunchConfig
from dlrover.python.tests.test_utils import start_local_master
from dlrover.trainer.torch.elastic_run import (
    _check_dlrover_master_available,
    _check_to_use_dlrover_run,
    _elastic_config_from_args,
    _launch_dlrover_local_master,
    parse_args,
    run,
    wait_pre_check,
    ElasticLaunch,
    _setup_dynamic_failover_extension,
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
        MasterClient._instance = None

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
            _check_to_use_dlrover_run(job_name)
            self.fail("Expected an Exception but got none.")
        except Exception:
            pass

        # 1) no dist master 2) standalone mode + no local master 3) node 0
        mock_check_dlrover_master.return_value = False
        mock_launch_local_master.return_value = None, "127.0.0.1:8000"
        use_dlrover_launch, master_handler = _check_to_use_dlrover_run(
            job_name, True
        )
        self.assertFalse(use_dlrover_launch)
        self.assertEqual(master_handler, None)

        # 1) no dist master 2) standalone mode + local master 3) node 0
        mock_check_dlrover_master.side_effect = [False, True]
        mock_launch_local_master.return_value = (
            "mock_process",
            "127.0.0.1:8000",
        )
        use_dlrover_launch, master_handler = _check_to_use_dlrover_run(
            job_name, True
        )
        self.assertTrue(use_dlrover_launch)
        self.assertIsNotNone(master_handler)

        # 1) with master address 2) node 0
        mock_check_dlrover_master.side_effect = None
        mock_check_dlrover_master.return_value = True
        use_dlrover_launch, master_handler = _check_to_use_dlrover_run(
            job_name, True
        )
        self.assertTrue(use_dlrover_launch)
        self.assertEqual(master_handler, None)

        # 1) with master address 2) node 1
        env_utils.set_env(NodeEnv.WORKER_RANK, "1")
        use_dlrover_launch, master_handler = _check_to_use_dlrover_run(
            job_name, True
        )
        self.assertTrue(use_dlrover_launch)

        # 1) no master address 2) node 1
        mock_check_dlrover_master.return_value = False
        try:
            _check_to_use_dlrover_run(job_name, True)
            self.fail("Expected an Exception but got none.")
        except Exception:
            pass

        # 1) no dist master 2) standalone mode 3) node 1
        try:
            _check_to_use_dlrover_run(job_name)
            self.fail("Expected an Exception but got none.")
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
            "--numa-affinity",
            "--membind-policy",
            "preferred",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        elastic = ElasticLaunch(
            config=config,
            entrypoint=cmd,
            use_dlrover_launch=True,
        )

        self.assertEqual(config.precheck, 1)
        self.assertTrue(config.network_check)
        self.assertFalse(config.comm_perf_test)
        self.assertTrue(config.auto_tunning)
        self.assertEqual(config.node_unit, 4)
        self.assertEqual(config.rdzv_configs["node_unit"], 4)
        self.assertEqual(config.training_port, 1000)
        self.assertTrue("bin/python" in cmd)
        self.assertListEqual(cmd_args, ["-u", "test.py", "--batch_size", "16"])
        self.assertTrue(config.numa_affinity)
        self.assertEqual(config.membind_policy, "preferred")
        self.assertTrue(elastic._config.numa_affinity)
        self.assertFalse("dlrover_run_affinity.sh" in elastic._entrypoint)

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

    @patch("dlrover.trainer.torch.elastic_run.ElasticLaunch")
    @patch("dlrover.trainer.torch.elastic_run._elastic_config_from_args")
    @patch("dlrover.trainer.torch.elastic_run._check_to_use_dlrover_run")
    def test_run(
        self,
        mock_check_dlrover,
        mock_elastic_config_from_args,
        mock_elastic_launch,
    ):
        test_cases = [
            {
                "name": "standalone_without_dlrover",
                "args": Namespace(standalone=True, other_arg="value"),
                "check_dlrover_return": (False, None),
                "use_dlrover_launch": False,
                "verify_rdzv": True,
                "master_handler": None,
            },
            {
                "name": "standalone_with_dlrover",
                "args": Namespace(standalone=True, other_arg="value"),
                "use_dlrover_launch": True,
                "verify_rdzv": False,
            },
            {
                "name": "distributed_mode",
                "args": Namespace(standalone=False, other_arg="value"),
                "check_dlrover_return": (True, None),
                "use_dlrover_launch": True,
                "verify_rdzv": False,
                "master_handler": None,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                # Reset all mocks before each test case
                mock_check_dlrover.reset_mock()
                mock_elastic_config_from_args.reset_mock()
                mock_elastic_launch.reset_mock()

                if case["name"] == "standalone_with_dlrover":
                    master_handler = MagicMock()
                    master_handler.close = MagicMock()
                    case["check_dlrover_return"] = (True, master_handler)
                else:
                    if case["master_handler"]:
                        case["master_handler"].reset_mock()

                mock_check_dlrover.return_value = case["check_dlrover_return"]

                mock_config = MagicMock()
                mock_cmd = MagicMock()
                mock_cmd_args = []
                mock_elastic_config_from_args.return_value = (
                    mock_config,
                    mock_cmd,
                    mock_cmd_args,
                )

                mock_instance = MagicMock()
                mock_elastic_launch.return_value = mock_instance

                run(case["args"])

                mock_check_dlrover.assert_called_once_with(
                    ANY, case["args"].standalone
                )

                if case["verify_rdzv"]:
                    self.assertEqual(case["args"].rdzv_backend, "c10d")
                    self.assertEqual(
                        case["args"].rdzv_endpoint, "localhost:29400"
                    )
                    self.assertTrue(hasattr(case["args"], "rdzv_id"))
                else:
                    self.assertFalse(hasattr(case["args"], "rdzv_backend"))

                mock_elastic_launch.assert_called_once_with(
                    config=mock_config,
                    entrypoint=mock_cmd,
                    use_dlrover_launch=case["use_dlrover_launch"],
                )
                mock_elastic_launch.return_value.assert_called_once_with(
                    *mock_cmd_args
                )

                if case["name"] == "standalone_with_dlrover":
                    master_handler.close.assert_called_once()


class TestMainFunction(unittest.TestCase):
    def setUp(self):
        self.patchers = []

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    def test_main_with_dlrover_master(self):
        env_patcher = patch(
            "dlrover.trainer.torch.compatible_main.env_utils.get_env",
            return_value="127.0.0.1:12345",
        )
        self.patchers.append(env_patcher)
        mock_get_env = env_patcher.start()

        dlrover_patcher = patch(
            "dlrover.trainer.torch.elastic_run.main",
            return_value=None,
        )
        self.patchers.append(dlrover_patcher)
        mock_dlrover_main = dlrover_patcher.start()
        torch_patcher = patch(
            "torch.distributed.run.main",
            return_value=None,
        )
        self.patchers.append(torch_patcher)
        mock_torch_main = torch_patcher.start()

        dlrover.trainer.torch.compatible_main.main()

        mock_get_env.assert_called_once_with(NodeEnv.DLROVER_MASTER_ADDR)
        mock_dlrover_main.assert_called_once()
        mock_torch_main.assert_not_called()

    def test_main_without_dlrover_master(self):
        env_patcher = patch(
            "dlrover.trainer.torch.compatible_main.env_utils.get_env",
            return_value=None,
        )
        self.patchers.append(env_patcher)
        mock_get_env = env_patcher.start()

        dlrover_patcher = patch(
            "dlrover.trainer.torch.elastic_run.main",
            return_value=None,
        )
        self.patchers.append(dlrover_patcher)
        mock_dlrover_main = dlrover_patcher.start()
        torch_patcher = patch(
            "torch.distributed.run.main",
            return_value=None,
        )
        self.patchers.append(torch_patcher)
        mock_torch_main = torch_patcher.start()

        dlrover.trainer.torch.compatible_main.main()

        mock_get_env.assert_called_once_with(NodeEnv.DLROVER_MASTER_ADDR)
        mock_dlrover_main.assert_not_called()
        mock_torch_main.assert_called_once()

    def test_setup_dynamic_failover_extension(self):
        config = ElasticLaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        # Test 1: No extension configured
        with patch.dict("os.environ", clear=True):
            _setup_dynamic_failover_extension(config)
            self.assertIsNone(config.dynamic_failover_extension)

        # Test 2: Valid extension configured
        extension_path = "elastic_run_test::TestDynamicAgentFailoverExtension"
        with patch.dict(
            "os.environ",
            {NodeEnv.DLROVER_EXTENSION_DYNAMIC_FAILOVER: extension_path},
        ):
            _setup_dynamic_failover_extension(config)
            self.assertIsNotNone(config.dynamic_failover_extension)
            self.assertTrue(
                isinstance(
                    config.dynamic_failover_extension,
                    DynamicAgentFailoverExtension,
                )
            )
            config.dynamic_failover_extension = None

        # Test 3: Invalid extension format
        with patch.dict(
            "os.environ",
            {NodeEnv.DLROVER_EXTENSION_DYNAMIC_FAILOVER: "invalid_format"},
        ):
            _setup_dynamic_failover_extension(config)
            self.assertIsNone(config.dynamic_failover_extension)

        # Test 4: Non-existent module
        with patch.dict(
            "os.environ",
            {
                NodeEnv.DLROVER_EXTENSION_DYNAMIC_FAILOVER: "nonexistent.module::Class"
            },
        ):
            _setup_dynamic_failover_extension(config)
            self.assertIsNone(config.dynamic_failover_extension)


class TestDynamicAgentFailoverExtension(DynamicAgentFailoverExtension):
    def get_user_failover_strategy(
        self, failure_info: AgentFailureInfo
    ) -> FailoverStrategy:
        return FailoverStrategy.ABORTION_FAILOVER
