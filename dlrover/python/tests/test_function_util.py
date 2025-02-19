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

import dlrover.python.util.function_util as fu


class FunctionUtilTest(unittest.TestCase):
    def test_timeout_decorator(self):
        @fu.timeout(1 * 1)
        def test(seconds):
            import time

            time.sleep(seconds)
            return

        try:
            test(2)
            self.fail()
        except fu.TimeoutException:
            pass

        def get_timeout():
            return 1

        @fu.timeout(callback_func=get_timeout)
        def test(seconds):
            import time

            time.sleep(seconds)
            return

        try:
            test(2)
            self.fail()
        except fu.TimeoutException:
            pass
