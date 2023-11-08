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
import unittest

from dlrover.python.common.global_context import Context


class GlobalContextTest(unittest.TestCase):
    def test_config_master_port(self):
        ctx = Context.singleton_instance()
        ctx.config_master_port(50001)
        self.assertEqual(ctx.master_port, 50001)
        os.environ["HOST_PORTS"] = "20000,20001,20002,20003"
        ctx.config_master_port(0)
        self.assertTrue(ctx.master_port in [20000, 20001, 20002, 20003])
        ctx.master_port = None
        os.environ["HOST_PORTS"] = ""
        ctx.config_master_port(0)
        self.assertTrue(ctx.master_port > 20000)


if __name__ == "__main__":
    unittest.main()
