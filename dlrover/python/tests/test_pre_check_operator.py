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

from dlrover.python.diagnosis.common.diagnosis_action import NoAction
from dlrover.python.master.diagnosis.precheck_operator import (
    NoPreCheckOperator,
)


class PreCheckOperatorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_no_pre_check_op(self):
        op = NoPreCheckOperator()
        self.assertTrue(op.check())
        self.assertTrue(isinstance(op.recover_actions()[0], NoAction))
        self.assertEqual(op.get_retry_interval_secs(), 5)
        self.assertEqual(op.get_retry_times(), 3)
        self.assertTrue(isinstance(op.failed_actions()[0], NoAction))


if __name__ == "__main__":
    unittest.main()
