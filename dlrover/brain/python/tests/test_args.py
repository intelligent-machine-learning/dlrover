# Copyright 2026 The DLRover Authors. All rights reserved.
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

import sys
import unittest
from unittest.mock import patch
from dlrover.brain.python.common.args import (
    get_parsed_args,
)


class TestArgParser(unittest.TestCase):
    def test_default_port(self):
        """Test that the default port is 8000 when no arguments are provided."""
        # Fake sys.argv as if the user just ran: python script.py
        test_args = ["script_name.py"]

        with patch.object(sys, "argv", test_args):
            args = get_parsed_args()
            self.assertEqual(args.port, 8000)

    def test_custom_port(self):
        """Test that a custom port is parsed correctly."""
        # Fake sys.argv as if the user ran: python script.py --port 9090
        test_args = ["script_name.py", "--port", "9090"]

        with patch.object(sys, "argv", test_args):
            args = get_parsed_args()
            self.assertEqual(args.port, 9090)

    @patch("dlrover.brain.python.common.args.logger")
    def test_unknown_arguments_logging(self, mock_logger):
        """Test that unknown arguments are gracefully ignored and logged."""
        # Fake sys.argv with a valid port AND some random unknown flags
        test_args = [
            "script_name.py",
            "--port",
            "8080",
            "--random-flag",
            "value",
        ]

        with patch.object(sys, "argv", test_args):
            args = get_parsed_args()

            # 1. It should still parse the valid port correctly
            self.assertEqual(args.port, 8080)

            # 2. It should log the unknown arguments exactly as you defined
            mock_logger.info.assert_called_once_with(
                "Unknown arguments: %s", ["--random-flag", "value"]
            )

    def test_invalid_port_type(self):
        """Test that passing a non-integer port raises an automatic SystemExit."""
        test_args = ["script_name.py", "--port", "not_a_number"]

        with patch.object(sys, "argv", test_args):
            # argparse automatically calls sys.exit(2) if a type check fails
            with self.assertRaises(SystemExit) as cm:
                get_parsed_args()

            # Verify the exit code is 2 (Standard argparse error code)
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
