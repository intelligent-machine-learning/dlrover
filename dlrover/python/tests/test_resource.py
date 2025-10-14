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

import unittest
from unittest.mock import patch, MagicMock

from dlrover.python.common.resource import Resource
from dlrover.python.elastic_agent.monitor.resource import (
    get_hpu_stats,
    get_gpu_stats,
)


class ResourceTest(unittest.TestCase):
    def test_basic(self):
        resource = Resource()
        self.assertEqual(resource.cpu, 0)
        self.assertEqual(resource.memory, 0)

        resource = Resource(cpu=2, gpu=1)
        self.assertEqual(resource.cpu, 2)
        self.assertEqual(resource.gpu, 1)
        self.assertEqual(resource.memory, 0)

        resource = Resource.default_gpu()
        self.assertEqual(resource.cpu, 0)
        self.assertEqual(resource.gpu, 1)
        self.assertEqual(resource.memory, 0)

        resource_dict = {
            "CPU": 1.1,
            "mem": 123,
            "Gpu": 1.0,
            "k1": "v1",
            "k2": "v2",
        }
        resource = Resource.from_dict(resource_dict)
        self.assertEqual(resource.cpu, 1.1)
        self.assertEqual(resource.gpu, 1.0)
        self.assertEqual(resource.memory, 123)
        self.assertEqual(resource.disk, 0)
        self.assertEqual(resource.ud_resource, {"k1": "v1", "k2": "v2"})
        self.assertTrue(resource.validate())
        self.assertEqual(resource.to_dict(cpu_flag="CPU").get("CPU"), 1.1)

        resource = Resource(cpu=-2, gpu=1)
        self.assertFalse(resource.validate())

    def test_get_gpu_stats_with_mock_pynvml(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock()

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            self.assertEqual(len(get_gpu_stats()), 2)

            mock_pynvml.nvmlDeviceGetCount.side_effect = Exception()
            self.assertEqual(get_hpu_stats(), [])

    def test_get_gpu_stats_without_acl(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            self.assertEqual(get_gpu_stats(), [])

    def test_get_hpu_stats_without_acl(self):
        self.assertEqual(get_hpu_stats(), [])

    def test_get_hpu_stats_with_mock_acl(self):
        mock_acl = MagicMock()
        mock_acl.init.return_value = None
        mock_acl.rt.get_device_count.return_value = 2, 0
        mock_acl.rt.get_mem_info.return_value = 100, 50, 0
        mock_acl.rt.get_device_utilization_rate.return_value = (
            {"cube_utilization": 0.3},
            0,
        )

        with patch.dict("sys.modules", {"acl": mock_acl}):
            result = get_hpu_stats()
            self.assertEqual(len(result), 2)
            mock_acl.init.assert_called_once()
            mock_acl.finalize.assert_called_once()

            mock_acl.rt.get_device_utilization_rate.side_effect = Exception()
            result = get_hpu_stats()
            self.assertEqual(len(result), 0)

            mock_acl.rt.get_device_count.side_effect = Exception()
            result = get_hpu_stats()
            self.assertEqual(len(result), 0)

            mock_acl.rt.get_mem_info.return_value = 0, 0, 1
            self.assertEqual(len(result), 0)

            mock_acl.rt.get_device_utilization_rate.return_value = ({}, 1)
            result = get_hpu_stats()
            self.assertEqual(len(result), 0)
