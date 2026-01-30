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
import tempfile
import shutil


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

    def test_logger_handlers_with_base_log_file(self):
        from dlrover.python.common.log import get_logger

        temp_dir = tempfile.mkdtemp()
        try:
            os.environ[BasicClass.LOG_ROOT_DIR_ENV] = temp_dir
            logger = get_logger("test_logger_with_file")

            self.assertTrue(len(logger.handlers) >= 2)

            has_file_handler = any(
                handler.__class__.__name__ == "RotatingFileHandler"
                for handler in logger.handlers
            )
            self.assertTrue(has_file_handler)

            expected_log_file = os.path.join(temp_dir, "dlrover.log")
            self.assertTrue(os.path.exists(expected_log_file))

            logger.info("test message for file logging")

            with open(expected_log_file, "r") as f:
                content = f.read()
                self.assertIn("test message for file logging", content)

        finally:
            shutil.rmtree(temp_dir)
            if BasicClass.LOG_ROOT_DIR_ENV in os.environ:
                del os.environ[BasicClass.LOG_ROOT_DIR_ENV]

    def test_logger_handlers_without_base_log_file(self):
        from dlrover.python.common.log import get_logger

        if BasicClass.LOG_ROOT_DIR_ENV in os.environ:
            del os.environ[BasicClass.LOG_ROOT_DIR_ENV]

        logger = get_logger("test_logger_without_file")

        self.assertTrue(len(logger.handlers) >= 1)

        has_file_handler = any(
            handler.__class__.__name__ == "RotatingFileHandler"
            for handler in logger.handlers
        )
        self.assertFalse(has_file_handler)

        logger.info("test message for console logging")

    def test_logger_handlers_empty_list(self):
        from dlrover.python.common.log import get_logger

        logger = get_logger("test_logger_empty_handlers", handlers=[])
        self.assertTrue(len(logger.handlers) >= 1)
        logger.info("test message with empty handlers")
