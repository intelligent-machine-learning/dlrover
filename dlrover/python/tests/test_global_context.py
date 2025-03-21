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
from dlrover.python.tests.test_diagnosis_master import TestOperator


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

    def test_get_pre_check_operators(self):
        ctx = Context.singleton_instance()
        ctx.pre_check_operators = []
        self.assertEqual(ctx.get_pre_check_operators(), [])
        self.assertFalse(ctx.pre_check_enabled())

        ctx.pre_check_operators = None
        self.assertEqual(ctx.get_pre_check_operators(), [])
        self.assertFalse(ctx.pre_check_enabled())

        ctx.pre_check_operators = ["o1"]
        self.assertEqual(ctx.get_pre_check_operators(), [])
        self.assertFalse(ctx.pre_check_enabled())

        ctx.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                True,
            )
        ]
        self.assertEqual(
            ctx.get_pre_check_operators()[0].__class__.__name__, "TestOperator"
        )
        self.assertTrue(ctx.pre_check_enabled())

    def test_is_pre_check_operator_bypass(self):
        ctx = Context.singleton_instance()
        ctx.pre_check_operators = []
        self.assertFalse(ctx.is_pre_check_operator_bypass(None))

        ctx.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                True,
            )
        ]
        self.assertFalse(ctx.is_pre_check_operator_bypass(None))
        self.assertTrue(ctx.is_pre_check_operator_bypass(TestOperator()))

        ctx.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                "y",
            )
        ]
        self.assertTrue(ctx.is_pre_check_operator_bypass(TestOperator()))

        ctx.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                False,
            )
        ]
        self.assertFalse(ctx.is_pre_check_operator_bypass(TestOperator()))

        ctx.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                "false",
            )
        ]
        self.assertFalse(ctx.is_pre_check_operator_bypass(TestOperator()))


if __name__ == "__main__":
    unittest.main()
