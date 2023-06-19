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

from dlrover.python.common.grpc import find_free_port, find_free_port_in_range


class GRPCUtilTest(unittest.TestCase):
    def test_find_free_port(self):
        port = find_free_port()
        self.assertTrue(port > 0)
        port = find_free_port_in_range(50001, 65535)
        self.assertTrue(port > 50000)


if __name__ == "__main__":
    unittest.main()
