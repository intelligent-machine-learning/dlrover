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

import unittest
from datetime import datetime, timedelta

import dlrover.python.util.time_util as tu


class TimeUtilTest(unittest.TestCase):
    def test_has_expired(self):
        self.assertFalse(
            tu.has_expired(
                (datetime.now() + timedelta(seconds=5)).timestamp(), 5
            )
        )
        self.assertTrue(
            tu.has_expired(
                (datetime.now() - timedelta(seconds=5)).timestamp(), 5
            )
        )

    def test_timestamp_diff_in_seconds(self):
        t1 = datetime.now()
        t2 = t1 + timedelta(seconds=5)
        self.assertEqual(
            tu.timestamp_diff_in_seconds(t1.timestamp(), t2.timestamp()), 5
        )
