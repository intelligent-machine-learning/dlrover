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

import os
import unittest

import pytest

from dlrover.python.common.constants import BasicClass


class LogTest(unittest.TestCase):
    @pytest.mark.run(order=1)
    def test_default_log_level(self):
        from dlrover.python.common.log import default_logger as logger
        from dlrover.python.common.log import get_log_level

        os.environ[BasicClass.LOG_LEVEL_ENV] = "INFO"
        self.assertEqual(get_log_level(), "INFO")
        logger.info("test123")

    @pytest.mark.run(order=2)
    def test_invalid_log_level(self):
        os.environ[BasicClass.LOG_LEVEL_ENV] = "INVALID"

        from dlrover.python.common.log import default_logger as logger
        from dlrover.python.common.log import get_log_level

        self.assertEqual(get_log_level(), "INFO")
        logger.info("test123")

    @pytest.mark.run(order=3)
    def test_debug_log_level(self):
        os.environ[BasicClass.LOG_LEVEL_ENV] = "DEBUG"

        from dlrover.python.common.log import default_logger as logger
        from dlrover.python.common.log import get_log_level

        self.assertEqual(get_log_level(), "DEBUG")
        logger.debug("test123456")
