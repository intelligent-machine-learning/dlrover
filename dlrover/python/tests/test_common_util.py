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

import socket
import unittest

import dlrover.python.util.common_util as cu


class CommonUtilTest(unittest.TestCase):
    def test_get_dlrover_version(self):
        self.assertIsNotNone(cu.get_dlrover_version())
        self.assertNotEqual(cu.get_dlrover_version(), "Unknown")

    def test_is_port_in_use(self):
        self.assertFalse(cu.is_port_in_use(65530))

    def test_find_free_port(self):
        port = cu.find_free_port()
        self.assertTrue(port > 0)
        port = cu.find_free_port_in_range(50001, 65535)
        self.assertTrue(port > 50000)

        port = cu.find_free_port_in_range(50001, 65535, False)
        self.assertTrue(port > 50000)

        ports = []
        for i in range(20):
            ports.append(20000 + i)
        port = cu.find_free_port_in_set(ports)
        self.assertTrue(port in ports)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 10000))
        with self.assertRaises(RuntimeError):
            cu.find_free_port_in_set([10000])
        with self.assertRaises(RuntimeError):
            cu.find_free_port_in_range(10000, 10000)
        s.close()

    def test_find_free_port_for_hccl(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 64003))
        port = cu.find_free_port_for_hccl()
        self.assertEqual(port, 64004)


if __name__ == "__main__":
    unittest.main()
