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

from unittest import mock
import unittest
from unittest.mock import MagicMock, patch

from dlrover.python.util.numa_util import (
    get_metaxgpu_affinity,
    get_metaxgpu_numa_node,
    get_metaxgpu_pci_bus,
)


class NUMAUtilTest(unittest.TestCase):
    def test_get_metaxgpu_pci_bus_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0000:00:1e.0"
        with patch("subprocess.run", return_value=mock_result):
            result = get_metaxgpu_pci_bus(0)
            self.assertEqual(result, "0000:00:1e.0")

    def test_get_metaxgpu_pci_bus_fail(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = get_metaxgpu_pci_bus(0)
            self.assertIsNone(result)

    def test_get_metaxgpu_numa_node_success(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_pci_bus",
            return_value="0000:00:1e.0",
        ):
            with patch(
                "builtins.open",
                MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(
                            read=MagicMock(return_value="1")
                        )
                    ),
                    __exit__=MagicMock(return_value=False),
                ),
            ):
                result = get_metaxgpu_numa_node(0)
                self.assertEqual(result, 1)

    def test_get_metaxgpu_numa_node_bus_none(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_pci_bus",
            return_value=None,
        ):
            result = get_metaxgpu_numa_node(0)
            self.assertIsNone(result)

    def test_get_metaxgpu_affinity_success(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_numa_node",
            return_value=0,
        ):
            with mock.patch(
                "builtins.open",
                mock.mock_open(read_data="0-3,8-11"),
            ):
                result = get_metaxgpu_affinity(0)
                self.assertEqual(result, {0, 1, 2, 3, 8, 9, 10, 11})

    def test_get_metaxgpu_affinity_node_none(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_numa_node",
            return_value=None,
        ):
            result = get_metaxgpu_affinity(0)
            self.assertIsNone(result)

    def test_get_metaxgpu_affinity_file_error(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_numa_node",
            return_value=1,
        ):
            with patch("builtins.open", side_effect=OSError("No such file")):
                result = get_metaxgpu_affinity(0)
                self.assertIsNone(result)

    def test_get_metaxgpu_affinity_value_error(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_numa_node",
            return_value=1,
        ):
            with patch(
                "builtins.open", side_effect=ValueError("No such value")
            ):
                result = get_metaxgpu_affinity(0)
                self.assertIsNone(result)

    def test_get_metaxgpu_affinity_exception(self):
        with patch(
            "dlrover.python.util.numa_util.get_metaxgpu_numa_node",
            return_value=1,
        ):
            with patch("builtins.open", side_effect=Exception("exception")):
                result = get_metaxgpu_affinity(0)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
