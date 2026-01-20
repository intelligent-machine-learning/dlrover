# Copyright 2022 The DLRover Authors. All rights reserved.
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
import subprocess
import tempfile
import unittest
from unittest import mock

import psutil

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv


class EnvUtilsTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(NodeEnv.NODE_ID, None)
        os.environ.pop(NodeEnv.NODE_RANK, None)
        os.environ.pop(NodeEnv.NODE_NUM, None)
        os.environ.pop(NodeEnv.WORKER_RANK, None)
        os.environ.pop("LOCAL_WORLD_SIZE", None)

    def test_get_env(self):
        os.environ[NodeEnv.WORKER_RANK] = "0"
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 0)

        os.environ[NodeEnv.WORKER_RANK] = "1"
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 1)

        os.environ[NodeEnv.NODE_RANK] = "2"
        node_rank = env_utils.get_node_rank()
        self.assertEqual(node_rank, 2)

        os.environ[NodeEnv.NODE_NUM] = "4"
        node_num = env_utils.get_node_num()
        self.assertEqual(node_num, 4)

        os.environ[NodeEnv.NODE_ID] = "1"
        node_id = env_utils.get_node_id()
        self.assertEqual(node_id, 1)

        node_type = env_utils.get_node_type()
        self.assertEqual(node_type, "worker")

        os.environ["LOCAL_WORLD_SIZE"] = "8"
        size = env_utils.get_local_world_size()
        self.assertEqual(size, 8)

    def test_get_hostname_and_ip(self):
        hostname, ipaddress = env_utils.get_hostname_and_ip()
        self.assertTrue(hostname)
        self.assertTrue(ipaddress)

    def test_get_kernel_stack(self):
        test_content = "kernel stack testing"

        # successful case
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=test_content)
            ):
                success, stack = env_utils.get_kernel_stack(12345)
                self.assertTrue(success)
                self.assertEqual(test_content, stack)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # FileNotFoundError
        with mock.patch("builtins.open", side_effect=FileNotFoundError()):
            success, stack = env_utils.get_kernel_stack(12345)
            self.assertFalse(success)
            self.assertIn("not exist", stack)

        # PermissionError
        with mock.patch("builtins.open", side_effect=PermissionError()):
            success, stack = env_utils.get_kernel_stack(12345)
            self.assertFalse(success)
            self.assertIn("permission denied", stack)

        # generic exception
        with mock.patch("builtins.open", side_effect=IOError("test error")):
            success, stack = env_utils.get_kernel_stack(12345)
            self.assertFalse(success)
            self.assertIn("unexpected error", stack)

    def test_get_user_stack_pyspy(self):
        test_content = "user stack testing"

        # successful case
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = test_content

        with mock.patch("subprocess.run", return_value=mock_result):
            success, stack = env_utils.get_user_stack_pyspy(12345)
            self.assertTrue(success)
            self.assertEqual(test_content, stack)

        # py-spy failure case
        mock_result.returncode = 1
        mock_result.stderr = "Failed to access process"

        with mock.patch("subprocess.run", return_value=mock_result):
            success, stack = env_utils.get_user_stack_pyspy(12345)
            self.assertFalse(success)
            self.assertIn("Failed to access process", stack)

        # timeout case
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("py-spy", 5),
        ):
            success, stack = env_utils.get_user_stack_pyspy(12345)
            self.assertFalse(success)
            self.assertIn("timed out", stack)

        # py-spy not installed case
        with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            success, stack = env_utils.get_user_stack_pyspy(12345)
            self.assertFalse(success)
            self.assertIn("not installed", stack)

        # generic exception
        with mock.patch(
            "subprocess.run", side_effect=RuntimeError("test error")
        ):
            success, stack = env_utils.get_user_stack_pyspy(12345)
            self.assertFalse(success)
            self.assertIn("unexpected error", stack)

    def test_get_all_child_pids(self):
        mock_parent_process = mock.MagicMock()
        mock_child1 = mock.MagicMock()
        mock_child1.pid = 1001
        mock_child2 = mock.MagicMock()
        mock_child2.pid = 1002
        mock_parent_process.children.return_value = [mock_child1, mock_child2]

        with mock.patch("psutil.Process", return_value=mock_parent_process):
            child_pids = env_utils.get_all_child_pids(1000)
            self.assertEqual(child_pids, [1001, 1002])
            mock_parent_process.children.assert_called_once_with(
                recursive=False
            )

        # no children case
        mock_parent_process.children.return_value = []
        with mock.patch("psutil.Process", return_value=mock_parent_process):
            child_pids = env_utils.get_all_child_pids(1000)
            self.assertEqual(child_pids, [])

        # NoSuchProcess exception
        with mock.patch(
            "psutil.Process",
            side_effect=psutil.NoSuchProcess("No such process"),
        ):
            child_pids = env_utils.get_all_child_pids(9999)
            self.assertEqual(child_pids, [])

        # generic exception
        with mock.patch("psutil.Process", side_effect=Exception("test error")):
            child_pids = env_utils.get_all_child_pids(1000)
            self.assertEqual(child_pids, [])

        # default parent_pid (current process)
        with mock.patch("psutil.Process") as mock_process_class:
            with mock.patch("os.getpid", return_value=5000):
                mock_process_class.return_value.children.return_value = []
                child_pids = env_utils.get_all_child_pids()
                self.assertEqual(child_pids, [])
                mock_process_class.assert_called_once_with(5000)
