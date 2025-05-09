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

import unittest

from dlrover.python.common.resource import Resource


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
